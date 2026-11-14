use std::path::{Path, PathBuf};

use ndarray::{Array4, ArrayD};
use ort::{
    session::builder::GraphOptimizationLevel,
    session::Session,
    value::{Outlet, TensorElementType, TensorRef},
};

#[cfg(feature = "directml")]
use crate::config::ExecutionProvider;
use crate::error::{
    ensure_model as ensure, ModelErrorKind, OcrError, OcrResult as Result, ResultExt,
    RuntimeErrorKind,
};
use crate::{cancellation::OcrCancellationToken, config::InferenceOptions, OcrRuntime};

pub(crate) struct OnnxSession {
    session: Session,
    input_shape: Vec<Option<usize>>,
    output_shape: Vec<Option<usize>>,
    model_path: PathBuf,
}

impl OnnxSession {
    pub(crate) fn new(
        model_path: impl AsRef<Path>,
        options: InferenceOptions,
        input_shape: [Option<usize>; 4],
        output_shape: &[Option<usize>],
        runtime: Option<&OcrRuntime>,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let metadata = std::fs::metadata(model_path).map_err(|source| {
            let (kind, message) = if source.kind() == std::io::ErrorKind::NotFound {
                (ModelErrorKind::NotFound, format!("ONNX model file not found at {}; check the config model path or prepare the model cache", model_path.display()))
            } else {
                (ModelErrorKind::Io, format!("failed to inspect ONNX model {}", model_path.display()))
            };
            OcrError::model(kind, Some(model_path), message).with_source(source)
        })?;
        if !metadata.is_file() {
            return Err(OcrError::model(
                ModelErrorKind::Io,
                Some(model_path),
                format!("ONNX model path is not a file: {}", model_path.display()),
            ));
        }
        options.validate()?;
        let runtime = match runtime {
            Some(runtime) => runtime.clone(),
            None => OcrRuntime::new()?,
        };
        let mut session_builder = Session::builder(runtime.environment())
            .map_err(session_configuration_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(session_configuration_error)?
            .with_intra_threads(options.intra_threads)
            .map_err(session_configuration_error)?
            .with_inter_threads(options.inter_threads)
            .map_err(session_configuration_error)?
            .with_parallel_execution(options.parallel_execution)
            .map_err(session_configuration_error)?
            .with_memory_pattern(
                input_shape.iter().all(Option::is_some)
                    && options.execution_provider != crate::config::ExecutionProvider::DirectMl,
            )
            .map_err(session_configuration_error)?;

        let mut providers = Vec::with_capacity(2);
        #[cfg(feature = "directml")]
        if options.execution_provider == ExecutionProvider::DirectMl {
            providers.push(ort::ep::DirectML::default().build().error_on_failure());
        }
        // Configure the CPU provider explicitly even for DirectML fallback so
        // enable_cpu_mem_arena has the same meaning in both execution modes.
        providers.push(
            ort::ep::CPU::default()
                .with_arena_allocator(options.enable_cpu_mem_arena)
                .build(),
        );
        session_builder = session_builder
            .with_execution_providers(providers)
            .map_err(session_configuration_error)?;

        let session = session_builder
            .commit_from_file(model_path)
            .map_err(|source| model_load_error(model_path, source))?;
        if session.inputs().len() != 1 {
            return Err(OcrError::model(
                ModelErrorKind::Incompatible,
                Some(model_path),
                format!(
                    "OCR model {} must have exactly one image input, got {}",
                    model_path.display(),
                    session.inputs().len()
                ),
            ));
        }
        let input_shape = validate_model_tensor(&session.inputs()[0], &input_shape, "input")
            .map_err(|error| error.with_model_path(model_path))
            .with_context(|| format!("incompatible ONNX model {}", model_path.display()))?;
        let output = session.outputs().first().ok_or_else(|| {
            OcrError::model(
                ModelErrorKind::Incompatible,
                Some(model_path),
                format!("ONNX model {} has no outputs", model_path.display()),
            )
        })?;
        let output_shape = validate_model_tensor(output, output_shape, "output")
            .map_err(|error| error.with_model_path(model_path))
            .with_context(|| format!("incompatible ONNX model {}", model_path.display()))?;
        Ok(Self {
            session,
            input_shape,
            output_shape,
            model_path: model_path.to_path_buf(),
        })
    }

    pub(crate) fn run_f32(
        &mut self,
        input: &Array4<f32>,
        cancellation: &OcrCancellationToken,
    ) -> Result<ArrayD<f32>> {
        cancellation.checkpoint()?;
        validate_tensor_shape(input.shape(), &self.input_shape, "input")
            .map_err(|error| error.with_model_path(&self.model_path))?;
        let run_options = cancellation.begin_onnx_run()?;
        let output = {
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![TensorRef::from_array_view(input).map_err(|source| {
                        OcrError::runtime(
                            RuntimeErrorKind::Execution,
                            "failed to create ONNX input tensor",
                        )
                        .with_source(source)
                    })?],
                    run_options.options(),
                )
                .map_err(|e| {
                    if cancellation.is_cancelled() {
                        crate::cancellation::OcrCancelled.into()
                    } else {
                        OcrError::runtime(RuntimeErrorKind::Execution, "ONNX inference failed")
                            .with_source(e)
                    }
                })?;
            let value = outputs.values().next().ok_or_else(|| {
                OcrError::model(
                    ModelErrorKind::Incompatible,
                    Some(&self.model_path),
                    "ONNX model returned no outputs",
                )
            })?;
            let view = value.try_extract_array::<f32>().map_err(|source| {
                OcrError::model(
                    ModelErrorKind::Incompatible,
                    Some(&self.model_path),
                    "ONNX model output must be a float32 tensor",
                )
                .with_source(source)
            })?;
            validate_prediction_shape(view.shape(), &self.output_shape, input.shape()[0])
                .map_err(|error| error.with_model_path(&self.model_path))?;
            cancellation.checkpoint()?;
            view.to_owned()
        };
        drop(run_options);
        cancellation.checkpoint()?;
        Ok(output)
    }
}

fn validate_model_tensor(
    outlet: &Outlet,
    expected: &[Option<usize>],
    label: &str,
) -> Result<Vec<Option<usize>>> {
    let dtype = outlet.dtype();
    ensure!(
        dtype.tensor_type() == Some(TensorElementType::Float32),
        "ONNX {label} '{}' must be a float32 tensor, got {dtype:?}",
        outlet.name()
    );
    let shape = dtype
        .tensor_shape()
        .ok_or_else(|| OcrError::incompatible_model("ONNX value is not a tensor"))?;
    ensure!(
        shape.len() == expected.len(),
        "ONNX {label} '{}' must have rank {}, got {shape}",
        outlet.name(),
        expected.len()
    );

    shape
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(axis, (actual, expected))| {
            ensure!(*actual != 0, "ONNX {label} dimension {axis} must not be zero");
            if *actual < 0 {
                return Ok(*expected);
            }
            let actual = usize::try_from(*actual).map_err(|source| OcrError::incompatible_model("ONNX dimension exceeds platform limits").with_source(source))?;
            if let Some(expected) = expected {
                ensure!(
                    actual == *expected,
                    "ONNX {label} dimension {axis} must be {expected}, got {actual}; check the model, image shape, dictionary, or labels"
                );
            }
            Ok(Some(actual))
        })
        .collect()
}

fn session_configuration_error(source: impl Into<ort::Error>) -> OcrError {
    OcrError::runtime(
        RuntimeErrorKind::SessionConfiguration,
        "failed to configure ONNX session",
    )
    .with_source(source.into())
}

fn model_load_error(path: &Path, source: ort::Error) -> OcrError {
    use ort::ErrorCode;
    let message = format!("failed to load ONNX model {}", path.display());
    let error = match source.code() {
        ErrorCode::NoSuchFile | ErrorCode::NoModel => {
            OcrError::model(ModelErrorKind::NotFound, Some(path), message)
        }
        ErrorCode::InvalidArgument | ErrorCode::InvalidProtobuf | ErrorCode::InvalidGraph => {
            OcrError::model(ModelErrorKind::Incompatible, Some(path), message)
        }
        _ => OcrError::runtime(RuntimeErrorKind::SessionConfiguration, message),
    };
    error.with_source(source)
}

fn validate_tensor_shape(actual: &[usize], expected: &[Option<usize>], label: &str) -> Result<()> {
    ensure!(
        actual.len() == expected.len(),
        "ONNX {label} tensor must have rank {}, got {actual:?}",
        expected.len()
    );
    for (axis, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            *actual > 0,
            "ONNX {label} dimension {axis} must be greater than zero"
        );
        if let Some(expected) = expected {
            ensure!(
                actual == expected,
                "ONNX {label} dimension {axis} must be {expected}, got {actual}"
            );
        }
    }
    Ok(())
}

fn validate_prediction_shape(
    actual: &[usize],
    expected: &[Option<usize>],
    batch_size: usize,
) -> Result<()> {
    validate_tensor_shape(actual, expected, "output")?;
    ensure!(
        actual[0] == batch_size,
        "ONNX output batch size must match input batch size {batch_size}, got {}",
        actual[0]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::value::{Shape, SymbolicDimensions, ValueType};

    fn float_tensor(shape: impl Into<Shape>) -> Outlet {
        let shape = shape.into();
        let dimension_symbols = SymbolicDimensions::empty(shape.len());
        Outlet::new(
            "test",
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape,
                dimension_symbols,
            },
        )
    }

    #[test]
    fn model_contract_accepts_dynamic_nchw_input() {
        let shape = validate_model_tensor(
            &float_tensor([-1_i64, 3, 48, -1]),
            &[None, Some(3), Some(48), None],
            "input",
        )
        .unwrap();
        assert_eq!(shape, [None, Some(3), Some(48), None]);
    }

    #[test]
    fn model_contract_rejects_wrong_channel_layout() {
        let error = validate_model_tensor(
            &float_tensor([1_i64, 48, 320, 3]),
            &[None, Some(3), Some(48), None],
            "input",
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("input dimension 1 must be 3, got 48"));
    }

    #[test]
    fn model_contract_rejects_dictionary_class_count_mismatch() {
        let error = validate_model_tensor(
            &float_tensor([-1_i64, -1, 4]),
            &[None, None, Some(5)],
            "output",
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("output dimension 2 must be 5, got 4"));
    }

    #[test]
    fn dynamic_output_cannot_change_batch_size() {
        let error = validate_prediction_shape(&[2, 80, 5], &[None, None, Some(5)], 1).unwrap_err();
        assert!(error
            .to_string()
            .contains("output batch size must match input batch size 1"));
    }
}
