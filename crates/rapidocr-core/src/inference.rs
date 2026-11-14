use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use ndarray::{Array4, ArrayD};
use ort::{
    session::builder::GraphOptimizationLevel,
    session::Session,
    value::{Outlet, TensorElementType, TensorRef},
};

#[cfg(feature = "directml")]
use crate::config::ExecutionProvider;
use crate::{cancellation::OcrCancellationToken, config::InferenceOptions, OcrRuntime};

pub(crate) struct OnnxSession {
    session: Session,
    input_shape: Vec<Option<usize>>,
    output_shape: Vec<Option<usize>>,
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
        if !model_path.exists() {
            bail!(
                "ONNX model file not found at {}; check the config model path or prepare the model cache",
                model_path.display()
            );
        }
        if !model_path.is_file() {
            bail!("ONNX model path is not a file: {}", model_path.display());
        }
        options.validate()?;
        let runtime = match runtime {
            Some(runtime) => runtime.clone(),
            None => OcrRuntime::new()?,
        };
        let mut session_builder = Session::builder(runtime.environment())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::<()>::from)?
            .with_intra_threads(options.intra_threads)
            .map_err(ort::Error::<()>::from)?
            .with_inter_threads(options.inter_threads)
            .map_err(ort::Error::<()>::from)?
            .with_parallel_execution(options.parallel_execution)
            .map_err(ort::Error::<()>::from)?
            .with_memory_pattern(
                input_shape.iter().all(Option::is_some)
                    && options.execution_provider != crate::config::ExecutionProvider::DirectMl,
            )
            .map_err(ort::Error::<()>::from)?;

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
            .map_err(ort::Error::<()>::from)?;

        let session = session_builder
            .commit_from_file(model_path)
            .with_context(|| format!("failed to load ONNX model {}", model_path.display()))?;
        ensure!(
            session.inputs().len() == 1,
            "OCR model {} must have exactly one image input, got {}",
            model_path.display(),
            session.inputs().len()
        );
        let input_shape = validate_model_tensor(&session.inputs()[0], &input_shape, "input")
            .with_context(|| format!("incompatible ONNX model {}", model_path.display()))?;
        let output = session
            .outputs()
            .first()
            .with_context(|| format!("ONNX model {} has no outputs", model_path.display()))?;
        let output_shape = validate_model_tensor(output, output_shape, "output")
            .with_context(|| format!("incompatible ONNX model {}", model_path.display()))?;
        Ok(Self {
            session,
            input_shape,
            output_shape,
        })
    }

    pub(crate) fn run_f32(
        &mut self,
        input: &Array4<f32>,
        cancellation: &OcrCancellationToken,
    ) -> Result<ArrayD<f32>> {
        cancellation.checkpoint()?;
        validate_tensor_shape(input.shape(), &self.input_shape, "input")?;
        let run_options = cancellation.begin_onnx_run()?;
        let output = {
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![TensorRef::from_array_view(input)?],
                    run_options.options(),
                )
                .map_err(|e| {
                    if cancellation.is_cancelled() {
                        crate::cancellation::OcrCancelled.into()
                    } else {
                        anyhow::Error::new(e)
                    }
                })?;
            let value = outputs
                .values()
                .next()
                .context("ONNX model returned no outputs")?;
            let view = value
                .try_extract_array::<f32>()
                .context("ONNX model output must be a float32 tensor")?;
            validate_prediction_shape(view.shape(), &self.output_shape, input.shape()[0])?;
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
    let shape = dtype.tensor_shape().context("ONNX value is not a tensor")?;
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
            let actual = usize::try_from(*actual).context("ONNX dimension exceeds platform limits")?;
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
