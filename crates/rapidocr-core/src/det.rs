use anyhow::{Context, Result};

use crate::{
    config::{DetConfig, LimitType},
    db_postprocess::{DbPostProcess, DbPostProcessConfig},
    image_ops::{resize_to_multiple_for_det, rgb_to_nchw},
    inference::OnnxSession,
    types::Quad,
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
        let input_img = resize_to_multiple_for_det(
            img,
            self.cfg.limit_side_len,
            matches!(self.cfg.limit_type, LimitType::Min),
        )?;
        let tensor = rgb_to_nchw(&input_img, self.cfg.mean, self.cfg.std);
        let pred = self.session.run_f32(&tensor)?;
        Ok(self
            .postprocess
            .process(pred, img.width(), img.height())?
            .into_iter()
            .map(|candidate| candidate.bbox)
            .collect())
    }
}
