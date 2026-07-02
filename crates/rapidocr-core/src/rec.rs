use std::{fs, path::Path, time::Instant};

use anyhow::{bail, Context, Result};
use image::{Rgb, RgbImage};
use ndarray::{s, Array4, Ix3};

use crate::{
    config::RecConfig,
    inference::OnnxSession,
    types::{OcrTimings, RecText},
};

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
        Ok(self.recognize_timed(imgs)?.texts)
    }

    pub fn recognize_timed(&mut self, imgs: &[RgbImage]) -> Result<RecognizeResult> {
        let mut timings = OcrTimings::default();
        if imgs.is_empty() {
            return Ok(RecognizeResult {
                texts: Vec::new(),
                timings,
            });
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

            let batch_w = (self.cfg.image_shape[1] as f32 * max_wh_ratio) as usize;
            let mut batch = Array4::<f32>::zeros((
                chunk.len(),
                self.cfg.image_shape[0],
                self.cfg.image_shape[1],
                batch_w,
            ));

            let start = Instant::now();
            for (i, img) in chunk.iter().enumerate() {
                let norm = self.resize_norm_img(img, max_wh_ratio)?;
                batch.slice_mut(s![i, .., .., ..]).assign(&norm);
            }
            timings.rec_preprocess_ms += elapsed_ms(start);

            let start = Instant::now();
            let pred = self.session.run_f32(&batch)?;
            timings.rec_inference_ms += elapsed_ms(start);

            let start = Instant::now();
            let pred = pred.into_dimensionality::<Ix3>()?;
            for i in 0..pred.shape()[0] {
                results.push(self.decode_one(pred.slice(s![i, .., ..]))?);
            }
            timings.rec_decode_ms += elapsed_ms(start);
        }
        Ok(RecognizeResult {
            texts: results,
            timings,
        })
    }

    fn resize_norm_img(&self, img: &RgbImage, max_wh_ratio: f32) -> Result<ndarray::Array3<f32>> {
        let [channels, img_h, _] = self.cfg.image_shape;
        if channels != 3 {
            bail!("only 3-channel recognition input is supported");
        }
        let img_w = (img_h as f32 * max_wh_ratio) as usize;
        let ratio = img.width() as f32 / img.height().max(1) as f32;
        let resized_w = ((img_h as f32 * ratio).ceil() as usize).min(img_w).max(1);
        let resized = resize_linear_opencv(img, resized_w as u32, img_h as u32);

        let mut out = ndarray::Array3::<f32>::zeros((channels, img_h, img_w));
        for (x, y, pixel) in resized.enumerate_pixels() {
            for c in 0..3 {
                out[[c, y as usize, x as usize]] = pixel[2 - c] as f32 / 255.0 / 0.5 - 1.0;
            }
        }
        Ok(out)
    }

    fn decode_one(&self, logits: ndarray::ArrayView2<'_, f32>) -> Result<RecText> {
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

pub struct RecognizeResult {
    pub texts: Vec<RecText>,
    pub timings: OcrTimings,
}

fn resize_linear_opencv(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    let src_w = img.width();
    let src_h = img.height();
    let mut out = RgbImage::new(width, height);
    let scale_x = src_w as f32 / width.max(1) as f32;
    let scale_y = src_h as f32 / height.max(1) as f32;

    for y in 0..height {
        let (y0, y1, wy) = linear_bounds(y, scale_y, src_h);
        for x in 0..width {
            let (x0, x1, wx) = linear_bounds(x, scale_x, src_w);
            let p00 = img.get_pixel(x0, y0).0;
            let p01 = img.get_pixel(x1, y0).0;
            let p10 = img.get_pixel(x0, y1).0;
            let p11 = img.get_pixel(x1, y1).0;

            let mut pixel = [0u8; 3];
            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - wx) + p01[c] as f32 * wx;
                let bottom = p10[c] as f32 * (1.0 - wx) + p11[c] as f32 * wx;
                pixel[c] = (top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x, y, Rgb(pixel));
        }
    }

    out
}

fn linear_bounds(dst: u32, scale: f32, src_len: u32) -> (u32, u32, f32) {
    if src_len <= 1 {
        return (0, 0, 0.0);
    }

    let src = (dst as f32 + 0.5) * scale - 0.5;
    if src <= 0.0 {
        return (0, 0, 0.0);
    }

    let low = src.floor() as u32;
    if low >= src_len - 1 {
        return (src_len - 1, src_len - 1, 0.0);
    }

    (low, low + 1, src - low as f32)
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
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
