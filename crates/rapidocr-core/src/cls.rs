use anyhow::{bail, Context, Result};
use image::{imageops, RgbImage};
use ndarray::{s, Array4, Ix2};

use crate::{config::ClsConfig, inference::OnnxSession};

#[derive(Debug, Clone)]
pub struct ClsResult {
    pub label: String,
    pub score: f32,
}

pub struct TextClassifier {
    cfg: ClsConfig,
    session: OnnxSession,
}

impl TextClassifier {
    pub fn new(cfg: ClsConfig) -> Result<Self> {
        cfg.validate().context("invalid classification config")?;
        let session = OnnxSession::new(&cfg.model_path).with_context(|| {
            format!(
                "failed to load classification model {}",
                cfg.model_path.display()
            )
        })?;
        Ok(Self { cfg, session })
    }

    pub fn classify(&mut self, imgs: &[RgbImage]) -> Result<Vec<ClsResult>> {
        if imgs.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(imgs.len());
        for chunk in imgs.chunks(self.cfg.batch_size) {
            let [channels, img_h, img_w] = self.cfg.image_shape;
            if channels != 3 {
                bail!("only 3-channel classification input is supported");
            }

            let mut batch = Array4::<f32>::zeros((chunk.len(), channels, img_h, img_w));
            for (i, img) in chunk.iter().enumerate() {
                let norm = self.resize_norm_img(img)?;
                batch.slice_mut(s![i, .., .., ..]).assign(&norm);
            }

            let pred = self.session.run_f32(&batch)?;
            let pred = pred.into_dimensionality::<Ix2>()?;
            for row in pred.outer_iter() {
                let (idx, score) = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(idx, score)| (idx, *score))
                    .unwrap_or((0, 0.0));
                let label = self
                    .cfg
                    .labels
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| idx.to_string());
                results.push(ClsResult { label, score });
            }
        }
        Ok(results)
    }

    pub fn classify_and_rotate(&mut self, imgs: &[RgbImage]) -> Result<Vec<RgbImage>> {
        let cls = self.classify(imgs)?;
        Ok(imgs
            .iter()
            .zip(cls)
            .map(|(img, cls)| {
                if cls.label.contains("180") && cls.score > self.cfg.thresh {
                    imageops::rotate180(img)
                } else {
                    img.clone()
                }
            })
            .collect())
    }

    fn resize_norm_img(&self, img: &RgbImage) -> Result<ndarray::Array3<f32>> {
        let [channels, img_h, img_w] = self.cfg.image_shape;
        if channels != 3 {
            bail!("only 3-channel classification input is supported");
        }

        let ratio = img.width() as f32 / img.height().max(1) as f32;
        let resized_w = ((img_h as f32 * ratio).ceil() as usize).min(img_w).max(1);
        let resized = imageops::resize(
            img,
            resized_w as u32,
            img_h as u32,
            imageops::FilterType::Triangle,
        );

        let mut out = ndarray::Array3::<f32>::zeros((channels, img_h, img_w));
        for (x, y, pixel) in resized.enumerate_pixels() {
            for c in 0..3 {
                out[[c, y as usize, x as usize]] = (pixel[2 - c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        Ok(out)
    }
}
