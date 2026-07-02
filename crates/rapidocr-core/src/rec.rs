use std::{fs, path::Path};

use anyhow::{bail, Context, Result};
use image::{imageops, RgbImage};
use ndarray::{s, Array4, Ix3};

use crate::{config::RecConfig, inference::OnnxSession, types::RecText};

pub struct TextRecognizer {
    cfg: RecConfig,
    session: OnnxSession,
    characters: Vec<String>,
}

impl TextRecognizer {
    pub fn new(cfg: RecConfig) -> Result<Self> {
        cfg.validate().context("invalid recognition config")?;
        let session = OnnxSession::new(&cfg.model_path).with_context(|| {
            format!(
                "failed to load recognition model {}",
                cfg.model_path.display()
            )
        })?;
        let characters = read_character_file(&cfg.dict_path)?;
        Ok(Self {
            cfg,
            session,
            characters,
        })
    }

    pub fn recognize(&mut self, imgs: &[RgbImage]) -> Result<Vec<RecText>> {
        if imgs.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(imgs.len());
        for chunk in imgs.chunks(self.cfg.batch_size) {
            let max_wh_ratio = chunk
                .iter()
                .map(|img| img.width() as f32 / img.height().max(1) as f32)
                .fold(
                    self.cfg.image_shape[2] as f32 / self.cfg.image_shape[1] as f32,
                    f32::max,
                );

            let mut batch = Array4::<f32>::zeros((
                chunk.len(),
                self.cfg.image_shape[0],
                self.cfg.image_shape[1],
                (self.cfg.image_shape[1] as f32 * max_wh_ratio).ceil() as usize,
            ));

            for (i, img) in chunk.iter().enumerate() {
                let norm = self.resize_norm_img(img, max_wh_ratio)?;
                batch.slice_mut(s![i, .., .., ..]).assign(&norm);
            }

            let pred = self.session.run_f32(&batch)?;
            let pred = pred.into_dimensionality::<Ix3>()?;
            for i in 0..pred.shape()[0] {
                results.push(self.decode_one(pred.slice(s![i, .., ..]).to_owned())?);
            }
        }
        Ok(results)
    }

    fn resize_norm_img(&self, img: &RgbImage, max_wh_ratio: f32) -> Result<ndarray::Array3<f32>> {
        let [channels, img_h, _] = self.cfg.image_shape;
        if channels != 3 {
            bail!("only 3-channel recognition input is supported");
        }
        let img_w = (img_h as f32 * max_wh_ratio).ceil() as usize;
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
                out[[c, y as usize, x as usize]] = pixel[c] as f32 / 255.0 / 0.5 - 1.0;
            }
        }
        Ok(out)
    }

    fn decode_one(&self, logits: ndarray::Array2<f32>) -> Result<RecText> {
        let mut text = String::new();
        let mut confs = Vec::new();
        let mut last_idx = usize::MAX;

        for timestep in logits.outer_iter() {
            let (idx, prob) = timestep
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, prob)| (idx, *prob))
                .unwrap_or((0, 0.0));

            if idx == 0 || idx == last_idx {
                last_idx = idx;
                continue;
            }

            if let Some(ch) = self.characters.get(idx) {
                text.push_str(ch);
                confs.push(prob);
            }
            last_idx = idx;
        }

        let score = if confs.is_empty() {
            0.0
        } else {
            confs.iter().sum::<f32>() / confs.len() as f32
        };

        Ok(RecText { text, score })
    }
}

fn read_character_file(path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read recognition dictionary {}", path.display()))?;
    if content.trim().is_empty() {
        bail!("recognition dictionary {} is empty", path.display());
    }
    let mut chars = Vec::new();
    chars.push("blank".to_string());
    chars.extend(content.lines().map(|line| line.trim_end().to_string()));
    chars.push(" ".to_string());
    Ok(chars)
}
