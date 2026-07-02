pub mod cls;
pub mod config;
pub mod db_postprocess;
pub mod det;
pub mod geometry;
pub mod image_ops;
pub mod inference;
pub mod model;
pub mod rec;
pub mod types;

#[cfg(test)]
mod e2e_tests;

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::{
    cls::TextClassifier,
    config::{PipelineConfig, RapidOcrConfig},
    det::TextDetector,
    image_ops::{
        apply_vertical_padding, crop_perspective, load_rgb_image, resize_image_within_bounds,
    },
    rec::TextRecognizer,
    types::{OcrLine, OcrOutput, Quad},
};

pub struct RapidOcr {
    cfg: RapidOcrConfig,
    detector: Option<TextDetector>,
    classifier: Option<TextClassifier>,
    recognizer: Option<TextRecognizer>,
}

impl RapidOcr {
    pub fn new(cfg: RapidOcrConfig) -> Result<Self> {
        cfg.validate().context("invalid OCR config")?;
        let detector = if cfg.pipeline.use_det {
            let det_cfg = cfg
                .det
                .clone()
                .context("pipeline.use_det is true but [det] config is missing")?;
            Some(TextDetector::new(det_cfg).context("failed to initialize detection stage")?)
        } else {
            None
        };
        let classifier = if cfg.pipeline.use_cls {
            let cls_cfg = cfg
                .cls
                .clone()
                .context("pipeline.use_cls is true but [cls] config is missing")?;
            Some(
                TextClassifier::new(cls_cfg)
                    .context("failed to initialize classification stage")?,
            )
        } else {
            None
        };
        let recognizer = if cfg.pipeline.use_rec {
            let rec_cfg = cfg
                .rec
                .clone()
                .context("pipeline.use_rec is true but [rec] config is missing")?;
            Some(TextRecognizer::new(rec_cfg).context("failed to initialize recognition stage")?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            detector,
            classifier,
            recognizer,
        })
    }

    pub fn from_config(cfg: RapidOcrConfig) -> Result<Self> {
        Self::new(cfg)
    }

    pub fn config(&self) -> &RapidOcrConfig {
        &self.cfg
    }

    pub fn pipeline(&self) -> PipelineConfig {
        self.cfg.pipeline
    }

    pub fn run_path(&mut self, image_path: impl AsRef<Path>) -> Result<OcrOutput> {
        let image = load_rgb_image(image_path)?;
        self.run_image(&image)
    }

    pub fn run_image(&mut self, image: &image::RgbImage) -> Result<OcrOutput> {
        if image.width() == 0 || image.height() == 0 {
            bail!("invalid image: width and height must be greater than 0");
        }

        let needs_recognition = self.recognizer.is_some();
        let (boxes, crops) = if self.detector.is_some() {
            self.detect_and_crop(image, needs_recognition)?
        } else {
            let bbox = Quad::from_xyxy(
                0.0,
                0.0,
                image.width().saturating_sub(1) as f32,
                image.height().saturating_sub(1) as f32,
            );
            (vec![bbox], vec![image.clone()])
        };

        let Some(recognizer) = &mut self.recognizer else {
            let lines = boxes
                .into_iter()
                .map(|bbox| OcrLine {
                    bbox,
                    text: String::new(),
                    score: 0.0,
                })
                .collect();
            return Ok(OcrOutput { lines });
        };

        let crops = if let Some(classifier) = &mut self.classifier {
            classifier.classify_and_rotate(&crops)?
        } else {
            crops
        };
        let rec = recognizer.recognize(&crops)?;

        let lines = boxes
            .into_iter()
            .zip(rec)
            .filter_map(|(bbox, text)| {
                if text.text.trim().is_empty() || text.score < self.cfg.text_score {
                    return None;
                }
                Some(OcrLine {
                    bbox,
                    text: text.text,
                    score: text.score,
                })
            })
            .collect();

        Ok(OcrOutput { lines })
    }

    fn detect_and_crop(
        &mut self,
        image: &image::RgbImage,
        needs_crops: bool,
    ) -> Result<(Vec<Quad>, Vec<image::RgbImage>)> {
        let original_width = image.width();
        let original_height = image.height();
        let (resized, ratio_w, ratio_h) =
            resize_image_within_bounds(image, self.cfg.min_side_len, self.cfg.max_side_len)?;
        let (det_image, padding_top) =
            apply_vertical_padding(&resized, self.cfg.width_height_ratio, self.cfg.min_height)?;

        let detector = self
            .detector
            .as_mut()
            .context("detection stage is not enabled")?;
        let mut boxes = detector.detect(&det_image)?;
        let crops = if needs_crops {
            boxes
                .iter()
                .map(|b| crop_perspective(&det_image, b))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        for b in &mut boxes {
            if padding_top > 0 {
                for point in &mut b.points {
                    point[1] -= padding_top as f32;
                }
            }
            b.scale(ratio_w, ratio_h);
            b.clip(original_width, original_height);
        }

        Ok((boxes, crops))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_reports_missing_detection_model_with_path() {
        let root = std::env::temp_dir().join(format!(
            "rapidocr-rs-missing-api-model-{}",
            std::process::id()
        ));
        let cfg = RapidOcrConfig::ppocr_v6_small(&root);

        let err = match RapidOcr::new(cfg) {
            Ok(_) => panic!("RapidOcr::new unexpectedly succeeded with a missing model"),
            Err(err) => err,
        };
        let message = format!("{err:#}");

        assert!(message.contains("failed to initialize detection stage"));
        assert!(message.contains("ONNX model file not found"));
        assert!(message.contains("PP-OCRv6_det_small.onnx"));
    }
}
