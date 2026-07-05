use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array4, ArrayD};
use ort::{session::builder::GraphOptimizationLevel, session::Session, value::TensorRef};

pub(crate) struct OnnxSession {
    session: Session,
}

impl OnnxSession {
    pub(crate) fn new(model_path: impl AsRef<Path>) -> Result<Self> {
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
        let session = Session::builder()
            .map_err(|e| anyhow!(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!(e.to_string()))?
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
