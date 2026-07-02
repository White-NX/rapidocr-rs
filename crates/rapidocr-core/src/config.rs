use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RapidOcrConfig {
    #[serde(default)]
    pub pipeline: PipelineConfig,
    pub text_score: f32,
    pub min_side_len: u32,
    pub max_side_len: u32,
    pub min_height: u32,
    pub width_height_ratio: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub det: Option<DetConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cls: Option<ClsConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rec: Option<RecConfig>,
}

impl RapidOcrConfig {
    pub fn ppocr_v6_small(model_dir: impl Into<PathBuf>) -> Self {
        crate::model::PPOCRV6_SMALL.config(model_dir)
    }

    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        Self::from_toml_str(&content)
            .with_context(|| format!("failed to load config {}", path.display()))
    }

    pub fn from_toml_str(content: &str) -> Result<Self> {
        let mut parsed: RapidOcrConfigFile =
            toml::from_str(content).context("failed to parse config TOML")?;

        if let Some(use_det) = parsed.use_det {
            parsed.cfg.pipeline.use_det = use_det;
        }
        if let Some(use_cls) = parsed.use_cls {
            parsed.cfg.pipeline.use_cls = use_cls;
        }
        if let Some(use_rec) = parsed.use_rec {
            parsed.cfg.pipeline.use_rec = use_rec;
        }

        parsed.cfg.validate()?;
        Ok(parsed.cfg)
    }

    pub fn write_toml_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        self.validate().context("invalid OCR config")?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create config dir {}", parent.display()))?;
        }
        let content = self.to_toml_string()?;
        fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn to_toml_string(&self) -> Result<String> {
        self.validate().context("invalid OCR config")?;
        toml::to_string_pretty(self).context("failed to serialize config")
    }

    pub fn validate(&self) -> Result<()> {
        self.pipeline.validate()?;
        ensure_unit_interval(self.text_score, "text_score")?;
        ensure!(self.min_side_len > 0, "min_side_len must be greater than 0");
        ensure!(self.max_side_len > 0, "max_side_len must be greater than 0");
        ensure!(
            self.min_side_len <= self.max_side_len,
            "min_side_len must be less than or equal to max_side_len"
        );
        ensure!(self.min_height > 0, "min_height must be greater than 0");
        ensure!(
            self.width_height_ratio == -1.0 || self.width_height_ratio > 0.0,
            "width_height_ratio must be positive or -1"
        );

        if self.pipeline.use_det {
            self.det
                .as_ref()
                .context("pipeline.use_det is true but [det] config is missing")?
                .validate()?;
        }
        if self.pipeline.use_cls {
            self.cls
                .as_ref()
                .context("pipeline.use_cls is true but [cls] config is missing")?
                .validate()?;
        }
        if self.pipeline.use_rec {
            self.rec
                .as_ref()
                .context("pipeline.use_rec is true but [rec] config is missing")?
                .validate()?;
        }

        Ok(())
    }

    pub fn with_pipeline(mut self, pipeline: PipelineConfig) -> Self {
        self.pipeline = pipeline;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub use_det: bool,
    pub use_cls: bool,
    pub use_rec: bool,
}

impl PipelineConfig {
    pub const fn full() -> Self {
        Self {
            use_det: true,
            use_cls: true,
            use_rec: true,
        }
    }

    pub const fn without_cls() -> Self {
        Self {
            use_det: true,
            use_cls: false,
            use_rec: true,
        }
    }

    pub const fn detection_only() -> Self {
        Self {
            use_det: true,
            use_cls: false,
            use_rec: false,
        }
    }

    pub const fn recognition_only() -> Self {
        Self {
            use_det: false,
            use_cls: false,
            use_rec: true,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !self.use_det && !self.use_rec {
            bail!("at least one of pipeline.use_det or pipeline.use_rec must be true");
        }
        if self.use_cls && !self.use_rec {
            bail!("pipeline.use_cls requires pipeline.use_rec because cls only rotates recognition crops");
        }
        Ok(())
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self::full()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetConfig {
    pub model_path: PathBuf,
    pub limit_side_len: u32,
    pub limit_type: LimitType,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f32,
    pub min_size: u32,
}

impl DetConfig {
    pub fn validate(&self) -> Result<()> {
        ensure_non_empty_path(&self.model_path, "det.model_path")?;
        ensure!(
            self.limit_side_len > 0,
            "det.limit_side_len must be greater than 0"
        );
        ensure_finite_array(self.mean, "det.mean")?;
        ensure_finite_array(self.std, "det.std")?;
        for (idx, value) in self.std.iter().enumerate() {
            ensure!(*value != 0.0, "det.std[{idx}] must not be 0");
        }
        ensure_unit_interval(self.thresh, "det.thresh")?;
        ensure_unit_interval(self.box_thresh, "det.box_thresh")?;
        ensure!(
            self.max_candidates > 0,
            "det.max_candidates must be greater than 0"
        );
        ensure!(
            self.unclip_ratio.is_finite() && self.unclip_ratio > 0.0,
            "det.unclip_ratio must be finite and greater than 0"
        );
        ensure!(self.min_size > 0, "det.min_size must be greater than 0");
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LimitType {
    Min,
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecConfig {
    pub model_path: PathBuf,
    pub dict_path: PathBuf,
    pub image_shape: [usize; 3],
    pub batch_size: usize,
}

impl RecConfig {
    pub fn validate(&self) -> Result<()> {
        ensure_non_empty_path(&self.model_path, "rec.model_path")?;
        ensure_non_empty_path(&self.dict_path, "rec.dict_path")?;
        ensure_image_shape(self.image_shape, "rec.image_shape")?;
        ensure!(self.batch_size > 0, "rec.batch_size must be greater than 0");
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClsConfig {
    pub model_path: PathBuf,
    pub image_shape: [usize; 3],
    pub batch_size: usize,
    pub thresh: f32,
    pub labels: Vec<String>,
}

impl ClsConfig {
    pub fn validate(&self) -> Result<()> {
        ensure_non_empty_path(&self.model_path, "cls.model_path")?;
        ensure_image_shape(self.image_shape, "cls.image_shape")?;
        ensure!(self.batch_size > 0, "cls.batch_size must be greater than 0");
        ensure_unit_interval(self.thresh, "cls.thresh")?;
        ensure!(!self.labels.is_empty(), "cls.labels must not be empty");
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct RapidOcrConfigFile {
    #[serde(flatten)]
    cfg: RapidOcrConfig,
    #[serde(default)]
    use_det: Option<bool>,
    #[serde(default)]
    use_cls: Option<bool>,
    #[serde(default)]
    use_rec: Option<bool>,
}

fn ensure_non_empty_path(path: &Path, label: &str) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{label} must not be empty");
    Ok(())
}

fn ensure_unit_interval(value: f32, label: &str) -> Result<()> {
    ensure!(
        value.is_finite() && (0.0..=1.0).contains(&value),
        "{label} must be finite and between 0 and 1"
    );
    Ok(())
}

fn ensure_finite_array(values: [f32; 3], label: &str) -> Result<()> {
    for (idx, value) in values.iter().enumerate() {
        ensure!(value.is_finite(), "{label}[{idx}] must be finite");
    }
    Ok(())
}

fn ensure_image_shape(shape: [usize; 3], label: &str) -> Result<()> {
    ensure!(shape[0] == 3, "{label}[0] must be 3 channels");
    ensure!(shape[1] > 0, "{label}[1] must be greater than 0");
    ensure!(shape[2] > 0, "{label}[2] must be greater than 0");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_toml_roundtrip_preserves_default_paths_and_flags() {
        let path = std::env::temp_dir().join(format!(
            "rapidocr-rs-config-roundtrip-{}.toml",
            std::process::id()
        ));
        let cfg = RapidOcrConfig::ppocr_v6_small("models");
        cfg.write_toml_file(&path).unwrap();

        let loaded = RapidOcrConfig::from_toml_file(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(loaded.pipeline, PipelineConfig::full());
        assert_eq!(loaded.min_height, 30);
        assert_eq!(loaded.width_height_ratio, 8.0);
        let det = loaded.det.as_ref().unwrap();
        let cls = loaded.cls.as_ref().unwrap();
        let rec = loaded.rec.as_ref().unwrap();
        assert_eq!(
            det.model_path,
            PathBuf::from("models/PP-OCRv6_det_small.onnx")
        );
        assert_eq!(
            cls.model_path,
            PathBuf::from("models/ch_ppocr_mobile_v2.0_cls_mobile.onnx")
        );
        assert_eq!(rec.dict_path, PathBuf::from("models/ppocrv6_dict.txt"));
        assert_eq!(cls.labels, vec!["0", "180"]);
    }

    #[test]
    fn legacy_top_level_pipeline_flags_are_loaded() {
        let mut content = RapidOcrConfig::ppocr_v6_small("models")
            .to_toml_string()
            .unwrap();
        content = content.replace(
            "[pipeline]\nuse_det = true\nuse_cls = true\nuse_rec = true\n\n",
            "",
        );
        content.insert_str(0, "use_det = false\nuse_cls = false\nuse_rec = true\n");

        let loaded = RapidOcrConfig::from_toml_str(&content).unwrap();

        assert_eq!(loaded.pipeline, PipelineConfig::recognition_only());
    }

    #[test]
    fn config_validation_rejects_cls_without_rec() {
        let cfg = RapidOcrConfig::ppocr_v6_small("models").with_pipeline(PipelineConfig {
            use_det: true,
            use_cls: true,
            use_rec: false,
        });

        let err = cfg.validate().unwrap_err().to_string();

        assert!(err.contains("pipeline.use_cls requires pipeline.use_rec"));
    }

    #[test]
    fn detection_only_config_does_not_require_recognition_assets() {
        let mut cfg = RapidOcrConfig::ppocr_v6_small("models")
            .with_pipeline(PipelineConfig::detection_only());
        cfg.cls = None;
        cfg.rec = None;

        cfg.validate().unwrap();
    }

    #[test]
    fn full_pipeline_requires_recognition_config() {
        let mut cfg = RapidOcrConfig::ppocr_v6_small("models");
        cfg.rec = None;

        let err = cfg.validate().unwrap_err().to_string();

        assert!(err.contains("pipeline.use_rec is true but [rec] config is missing"));
    }
}
