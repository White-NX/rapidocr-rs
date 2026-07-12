use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array4, ArrayD};
use ort::{session::builder::GraphOptimizationLevel, session::Session, value::TensorRef};

#[cfg(feature = "directml")]
use crate::config::ExecutionProvider;
use crate::config::InferenceOptions;

pub(crate) struct OnnxSession {
    session: Session,
}

impl OnnxSession {
    pub(crate) fn new(model_path: impl AsRef<Path>, options: InferenceOptions) -> Result<Self> {
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
        let mut session_builder = Session::builder()
            .map_err(|e| anyhow!(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!(e.to_string()))?
            .with_intra_threads(options.intra_threads)
            .map_err(|e| anyhow!(e.to_string()))?
            .with_inter_threads(options.inter_threads)
            .map_err(|e| anyhow!(e.to_string()))?
            .with_parallel_execution(options.parallel_execution)
            .map_err(|e| anyhow!(e.to_string()))?;

        #[cfg(feature = "directml")]
        if options.execution_provider == ExecutionProvider::DirectMl {
            session_builder = session_builder
                .with_memory_pattern(false)
                .map_err(|e| anyhow!(e.to_string()))?
                .with_execution_providers([ort::ep::DirectML::default().build().error_on_failure()])
                .map_err(|e| anyhow!(e.to_string()))?;
        }

        if options.execution_provider == crate::config::ExecutionProvider::Cpu {
            session_builder = session_builder
                .with_execution_providers([ort::ep::CPU::default()
                    .with_arena_allocator(options.enable_cpu_mem_arena)
                    .build()])
                .map_err(|e| anyhow!(e.to_string()))?;
        }

        let session = session_builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow!(e.to_string()))
            .with_context(|| format!("failed to load ONNX model {}", model_path.display()))?;
        Ok(Self { session })
    }

    pub(crate) fn run_f32(&mut self, input: &Array4<f32>) -> Result<ArrayD<f32>> {
        let outputs = self
            .session
            .run(ort::inputs![TensorRef::from_array_view(input)?])
            .map_err(|e| anyhow!(e.to_string()))?;
        Ok(outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| anyhow!(e.to_string()))?
            .to_owned())
    }
}
