use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

use crate::config::{ClsConfig, DetConfig, LimitType, PipelineConfig, RapidOcrConfig, RecConfig};

pub const PPOCRV6_DET_SMALL_URL: &str =
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx";
pub const PPOCRV6_DET_SMALL_SHA256: &str =
    "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f";
pub const PPOCRV6_REC_SMALL_URL: &str =
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx";
pub const PPOCRV6_REC_SMALL_SHA256: &str =
    "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884";
pub const PPOCRV4_CLS_URL: &str =
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx";
pub const PPOCRV4_CLS_SHA256: &str =
    "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c";
pub const PPOCRV6_DICT_URL: &str =
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelAssetKind {
    Detection,
    Classification,
    Recognition,
    Dictionary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelAssetSpec {
    pub name: &'static str,
    pub kind: ModelAssetKind,
    pub filename: &'static str,
    pub url: &'static str,
    pub sha256: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSetSpec {
    pub name: &'static str,
    pub det: ModelAssetSpec,
    pub cls: ModelAssetSpec,
    pub rec: ModelAssetSpec,
    pub dict: ModelAssetSpec,
}

impl ModelSetSpec {
    pub fn assets(&self) -> [ModelAssetSpec; 4] {
        [self.det, self.cls, self.rec, self.dict]
    }

    pub fn assets_for_pipeline(&self, pipeline: PipelineConfig) -> Vec<ModelAssetSpec> {
        let mut assets = Vec::new();
        if pipeline.use_det {
            assets.push(self.det);
        }
        if pipeline.use_cls {
            assets.push(self.cls);
        }
        if pipeline.use_rec {
            assets.push(self.rec);
            assets.push(self.dict);
        }
        assets
    }

    pub fn config(&self, model_dir: impl Into<PathBuf>) -> RapidOcrConfig {
        let model_dir = model_dir.into();
        RapidOcrConfig {
            pipeline: PipelineConfig::full(),
            text_score: 0.5,
            min_side_len: 30,
            max_side_len: 2000,
            min_height: 30,
            width_height_ratio: 8.0,
            det: Some(DetConfig {
                model_path: model_dir.join(self.det.filename),
                limit_side_len: 736,
                limit_type: LimitType::Min,
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
                thresh: 0.3,
                box_thresh: 0.5,
                max_candidates: 1000,
                unclip_ratio: 1.6,
                min_size: 3,
            }),
            cls: Some(ClsConfig {
                model_path: model_dir.join(self.cls.filename),
                image_shape: [3, 48, 192],
                batch_size: 6,
                thresh: 0.9,
                labels: vec!["0".to_string(), "180".to_string()],
            }),
            rec: Some(RecConfig {
                model_path: model_dir.join(self.rec.filename),
                dict_path: model_dir.join(self.dict.filename),
                image_shape: [3, 48, 320],
                batch_size: 6,
            }),
        }
    }
}

pub const PPOCRV6_SMALL: ModelSetSpec = ModelSetSpec {
    name: "ppocrv6-small",
    det: ModelAssetSpec {
        name: "PP-OCRv6 small detection model",
        kind: ModelAssetKind::Detection,
        filename: "PP-OCRv6_det_small.onnx",
        url: PPOCRV6_DET_SMALL_URL,
        sha256: Some(PPOCRV6_DET_SMALL_SHA256),
    },
    cls: ModelAssetSpec {
        name: "PP-OCRv4 text direction classifier",
        kind: ModelAssetKind::Classification,
        filename: "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        url: PPOCRV4_CLS_URL,
        sha256: Some(PPOCRV4_CLS_SHA256),
    },
    rec: ModelAssetSpec {
        name: "PP-OCRv6 small recognition model",
        kind: ModelAssetKind::Recognition,
        filename: "PP-OCRv6_rec_small.onnx",
        url: PPOCRV6_REC_SMALL_URL,
        sha256: Some(PPOCRV6_REC_SMALL_SHA256),
    },
    dict: ModelAssetSpec {
        name: "PP-OCRv6 recognition dictionary",
        kind: ModelAssetKind::Dictionary,
        filename: "ppocrv6_dict.txt",
        url: PPOCRV6_DICT_URL,
        sha256: None,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelDownloadMode {
    Missing,
    Never,
}

#[derive(Debug, Clone)]
pub struct ModelCache {
    root: PathBuf,
}

impl ModelCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn asset_path(&self, asset: ModelAssetSpec) -> PathBuf {
        self.root.join(asset.filename)
    }

    pub fn config_for(&self, model_set: &ModelSetSpec) -> RapidOcrConfig {
        model_set.config(&self.root)
    }

    pub fn missing_assets(&self, model_set: &ModelSetSpec) -> Vec<ModelAssetSpec> {
        model_set
            .assets()
            .into_iter()
            .filter(|asset| !self.asset_path(*asset).exists())
            .collect()
    }

    pub fn missing_assets_for_pipeline(
        &self,
        model_set: &ModelSetSpec,
        pipeline: PipelineConfig,
    ) -> Vec<ModelAssetSpec> {
        model_set
            .assets_for_pipeline(pipeline)
            .into_iter()
            .filter(|asset| !self.asset_path(*asset).exists())
            .collect()
    }

    pub fn ensure_model_set(
        &self,
        model_set: &ModelSetSpec,
        mode: ModelDownloadMode,
    ) -> Result<()> {
        self.ensure_assets(model_set.assets(), mode)
    }

    pub fn ensure_model_set_for_pipeline(
        &self,
        model_set: &ModelSetSpec,
        pipeline: PipelineConfig,
        mode: ModelDownloadMode,
    ) -> Result<()> {
        self.ensure_assets(model_set.assets_for_pipeline(pipeline), mode)
    }

    pub fn ensure_ppocrv6_small(&self, mode: ModelDownloadMode) -> Result<()> {
        self.ensure_model_set(&PPOCRV6_SMALL, mode)
    }

    fn ensure_assets(
        &self,
        assets: impl IntoIterator<Item = ModelAssetSpec>,
        mode: ModelDownloadMode,
    ) -> Result<()> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("failed to create model cache {}", self.root.display()))?;

        for asset in assets {
            self.ensure_asset(asset, mode)
                .with_context(|| format!("failed to prepare {}", asset.name))?;
        }
        Ok(())
    }

    fn ensure_asset(&self, asset: ModelAssetSpec, mode: ModelDownloadMode) -> Result<()> {
        let path = self.asset_path(asset);
        if path.exists() {
            if let Some(expected) = asset.sha256 {
                verify_sha256(&path, expected)?;
            }
            return Ok(());
        }

        if mode == ModelDownloadMode::Never {
            bail!(
                "missing model asset {} at {}; allow downloads or place the file in the model cache",
                asset.name,
                path.display()
            );
        }

        download_asset(asset, &path)
    }
}

pub fn ensure_ppocrv6_small_models(model_dir: impl AsRef<Path>) -> Result<PathBuf> {
    let cache = ModelCache::new(model_dir.as_ref());
    cache.ensure_ppocrv6_small(ModelDownloadMode::Missing)?;
    Ok(cache.root().to_path_buf())
}

fn download_asset(asset: ModelAssetSpec, path: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0 rapidocr-rs")
        .build()
        .context("failed to build HTTP client")?;
    let bytes = client
        .get(asset.url)
        .send()
        .with_context(|| format!("failed to download {} from {}", asset.name, asset.url))?
        .error_for_status()
        .with_context(|| format!("download returned an error for {}", asset.url))?
        .bytes()
        .with_context(|| format!("failed to read response body for {}", asset.url))?;

    let mut file = fs::File::create(path)
        .with_context(|| format!("failed to create model file {}", path.display()))?;
    file.write_all(&bytes)
        .with_context(|| format!("failed to write model file {}", path.display()))?;

    if let Some(expected) = asset.sha256 {
        verify_sha256(path, expected)?;
    }
    Ok(())
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read model file {}", path.display()))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        bail!(
            "sha256 mismatch for {}: expected {}, got {}",
            path.display(),
            expected,
            actual
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppocrv6_small_model_set_builds_default_config_paths() {
        let cfg = PPOCRV6_SMALL.config("models");

        assert_eq!(
            cfg.det.as_ref().unwrap().model_path,
            PathBuf::from("models/PP-OCRv6_det_small.onnx")
        );
        assert_eq!(
            cfg.cls.as_ref().unwrap().model_path,
            PathBuf::from("models/ch_ppocr_mobile_v2.0_cls_mobile.onnx")
        );
        assert_eq!(
            cfg.rec.as_ref().unwrap().dict_path,
            PathBuf::from("models/ppocrv6_dict.txt")
        );
    }

    #[test]
    fn model_set_selects_assets_for_pipeline_switches() {
        let without_cls = PPOCRV6_SMALL.assets_for_pipeline(PipelineConfig::without_cls());
        assert_eq!(without_cls.len(), 3);
        assert!(!without_cls
            .iter()
            .any(|asset| asset.kind == ModelAssetKind::Classification));

        let detection_only = PPOCRV6_SMALL.assets_for_pipeline(PipelineConfig::detection_only());
        assert_eq!(detection_only.len(), 1);
        assert_eq!(detection_only[0].kind, ModelAssetKind::Detection);

        let recognition_only =
            PPOCRV6_SMALL.assets_for_pipeline(PipelineConfig::recognition_only());
        assert_eq!(recognition_only.len(), 2);
        assert!(recognition_only
            .iter()
            .any(|asset| asset.kind == ModelAssetKind::Recognition));
        assert!(recognition_only
            .iter()
            .any(|asset| asset.kind == ModelAssetKind::Dictionary));
    }

    #[test]
    fn model_cache_reports_missing_assets_without_downloading() {
        let root = std::env::temp_dir().join(format!(
            "rapidocr-rs-missing-model-cache-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let cache = ModelCache::new(&root);

        let missing = cache.missing_assets(&PPOCRV6_SMALL);
        assert_eq!(missing.len(), 4);

        let err = cache
            .ensure_ppocrv6_small(ModelDownloadMode::Never)
            .unwrap_err();
        let message = format!("{err:#}");
        let _ = fs::remove_dir_all(&root);

        assert!(message.contains("failed to prepare PP-OCRv6 small detection model"));
        assert!(message.contains("missing model asset"));
    }
}
