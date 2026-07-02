use std::time::Instant;

use anyhow::{Context, Result};

use crate::{
    config::{DetConfig, LimitType},
    db_postprocess::{DbPostProcess, DbPostProcessConfig},
    image_ops::{resize_to_multiple_for_det, rgb_to_nchw},
    inference::OnnxSession,
    types::{OcrTimings, Quad},
};

pub struct TextDetector {
    cfg: DetConfig,
    session: OnnxSession,
    postprocess: DbPostProcess,
}

impl TextDetector {
    pub fn new(cfg: DetConfig) -> Result<Self> {
        cfg.validate().context("invalid detection config")?;
        let session = OnnxSession::new(&cfg.model_path).with_context(|| {
            format!(
                "failed to load detection model {}",
                cfg.model_path.display()
            )
        })?;
        let postprocess = DbPostProcess::new(DbPostProcessConfig::from(&cfg));
        Ok(Self {
            cfg,
            session,
            postprocess,
        })
    }

    pub fn detect(&mut self, img: &image::RgbImage) -> Result<Vec<Quad>> {
        Ok(self.detect_timed(img)?.boxes)
    }

    pub fn detect_timed(&mut self, img: &image::RgbImage) -> Result<DetectResult> {
        let mut timings = OcrTimings::default();

        let start = Instant::now();
        let input_img = resize_to_multiple_for_det(
            img,
            self.cfg.limit_side_len,
            matches!(self.cfg.limit_type, LimitType::Min),
        )?;
        let tensor = rgb_to_nchw(&input_img, self.cfg.mean, self.cfg.std);
        timings.det_preprocess_ms = elapsed_ms(start);

        let start = Instant::now();
        let pred = self.session.run_f32(&tensor)?;
        timings.det_inference_ms = elapsed_ms(start);

        let start = Instant::now();
        let boxes = self
            .postprocess
            .process(pred, img.width(), img.height())?
            .into_iter()
            .map(|candidate| candidate.bbox)
            .collect();
        timings.det_postprocess_ms = elapsed_ms(start);

        Ok(DetectResult { boxes, timings })
    }
}

pub struct DetectResult {
    pub boxes: Vec<Quad>,
    pub timings: OcrTimings,
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}
