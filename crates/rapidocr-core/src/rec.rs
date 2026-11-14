use std::{fs, path::Path, time::Instant};

use image::{Rgb, RgbImage};
use ndarray::{s, Array4, Ix3};

use crate::{
    cancellation::OcrCancellationToken,
    config::{InferenceOptions, RecConfig},
    error::{
        InputLimitKind, ModelErrorKind, OcrError, OcrResult as Result, OcrStage, ResultExt,
        RuntimeErrorKind,
    },
    inference::OnnxSession,
    types::{OcrTimings, RecText},
    OcrRuntime,
};

pub(crate) struct TextRecognizer {
    cfg: RecConfig,
    session: OnnxSession,
    characters: Vec<String>,
}

impl TextRecognizer {
    pub(crate) fn new(
        cfg: RecConfig,
        inference: InferenceOptions,
        runtime: Option<&OcrRuntime>,
    ) -> Result<Self> {
        cfg.validate().context("invalid recognition config")?;
        let characters = read_character_file(&cfg.dict_path)?;
        let session = OnnxSession::new(
            &cfg.model_path,
            inference,
            [None, Some(3), Some(cfg.image_shape[1]), None],
            &[None, None, Some(characters.len())],
            runtime,
        )
        .with_context(|| {
            format!(
                "failed to load recognition model {}",
                cfg.model_path.display()
            )
        })?;
        Ok(Self {
            cfg,
            session,
            characters,
        })
    }

    pub(crate) fn recognize_timed(
        &mut self,
        imgs: &[RgbImage],
        cancellation: &OcrCancellationToken,
    ) -> Result<RecognizeResult> {
        let mut timings = OcrTimings::default();
        cancellation.checkpoint()?;
        if imgs.is_empty() {
            return Ok(RecognizeResult {
                texts: Vec::new(),
                timings,
            });
        }

        // RapidOCR sorts crops by aspect ratio before batching. Grouping crops
        // with similar widths avoids padding every item to an unrelated wide
        // crop, then results are restored to detection order below.
        let mut indices = (0..imgs.len()).collect::<Vec<_>>();
        indices.sort_by(|a, b| {
            let a_ratio = imgs[*a].width() as f32 / imgs[*a].height().max(1) as f32;
            let b_ratio = imgs[*b].width() as f32 / imgs[*b].height().max(1) as f32;
            a_ratio.total_cmp(&b_ratio)
        });

        let mut results = (0..imgs.len()).map(|_| None).collect::<Vec<_>>();
        for chunk in indices.chunks(self.cfg.batch_size) {
            cancellation.checkpoint()?;
            let max_wh_ratio = chunk
                .iter()
                .map(|idx| imgs[*idx].width() as f32 / imgs[*idx].height().max(1) as f32)
                .fold(
                    self.cfg.image_shape[2] as f32 / self.cfg.image_shape[1] as f32,
                    f32::max,
                );

            // PaddleOCR recognition batches are padded to the widest crop in the
            // batch while preserving the configured input height.
            let batch_w = (self.cfg.image_shape[1] as f32 * max_wh_ratio) as usize;
            validate_batch_allocation(&self.cfg, chunk.len(), batch_w)?;
            let start = Instant::now();
            let mut batch = Array4::<f32>::zeros((
                chunk.len(),
                self.cfg.image_shape[0],
                self.cfg.image_shape[1],
                batch_w,
            ));

            for (i, idx) in chunk.iter().enumerate() {
                cancellation.checkpoint()?;
                let img = &imgs[*idx];
                let norm = self.resize_norm_img(img, max_wh_ratio, cancellation)?;
                batch.slice_mut(s![i, .., .., ..]).assign(&norm);
            }
            timings.rec_preprocess_ms += elapsed_ms(start);

            let start = Instant::now();
            let pred = self.session.run_f32(&batch, cancellation)?;
            timings.rec_inference_ms += elapsed_ms(start);

            let start = Instant::now();
            let pred = pred.into_dimensionality::<Ix3>().map_err(|source| {
                OcrError::incompatible_model("recognition output must have rank 3")
                    .with_model_path(&self.cfg.model_path)
                    .with_source(source)
            })?;
            for i in 0..pred.shape()[0] {
                cancellation.checkpoint()?;
                results[chunk[i]] = Some(self.decode_one(pred.slice(s![i, .., ..]))?);
            }
            timings.rec_decode_ms += elapsed_ms(start);
        }
        let results = results
            .into_iter()
            .enumerate()
            .map(|(idx, result)| {
                result.ok_or_else(|| {
                    OcrError::runtime(
                        RuntimeErrorKind::Execution,
                        format!("recognition result missing for crop {idx}"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RecognizeResult {
            texts: results,
            timings,
        })
    }

    fn resize_norm_img(
        &self,
        img: &RgbImage,
        max_wh_ratio: f32,
        cancellation: &OcrCancellationToken,
    ) -> Result<ndarray::Array3<f32>> {
        let [channels, img_h, _] = self.cfg.image_shape;
        if channels != 3 {
            return Err(OcrError::invalid_config(
                "only 3-channel recognition input is supported",
            ));
        }
        let img_w = (img_h as f32 * max_wh_ratio) as usize;
        let ratio = img.width() as f32 / img.height().max(1) as f32;
        let resized_w = ((img_h as f32 * ratio).ceil() as usize).min(img_w).max(1);
        let resized = resize_linear_opencv(img, resized_w as u32, img_h as u32, cancellation)?;

        let mut out = ndarray::Array3::<f32>::zeros((channels, img_h, img_w));
        for (x, y, pixel) in resized.enumerate_pixels() {
            for c in 0..3 {
                // Recognition models expect BGR channel order normalized to
                // [-1, 1], even though the crate stores images as RGB.
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
            if !prob.is_finite() {
                return Err(OcrError::incompatible_model(
                    "recognition output contains a non-finite confidence",
                )
                .with_model_path(&self.cfg.model_path));
            }

            // CTC decode: index 0 is blank and repeated non-blank labels collapse
            // into a single character.
            if idx == 0 || idx == last_idx {
                last_idx = idx;
                continue;
            }

            let ch = self.characters.get(idx).ok_or_else(|| {
                OcrError::incompatible_model(format!(
                    "recognition class {idx} is not present in the dictionary"
                ))
                .with_model_path(&self.cfg.dict_path)
            })?;
            text.push_str(ch);
            confs.push(prob);
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

pub(crate) struct RecognizeResult {
    pub(crate) texts: Vec<RecText>,
    pub(crate) timings: OcrTimings,
}

fn validate_batch_allocation(cfg: &RecConfig, batch_size: usize, width: usize) -> Result<()> {
    let [channels, height, _] = cfg.image_shape;
    let pixels = batch_size
        .checked_mul(height)
        .and_then(|size| size.checked_mul(width))
        .ok_or_else(|| {
            OcrError::input_limit(
                OcrStage::Recognition,
                InputLimitKind::Dimensions,
                None,
                Some(usize::MAX as u64),
                "recognition batch dimensions overflow platform limits",
            )
        })?;
    let bytes = pixels
        .checked_mul(channels)
        .and_then(|size| size.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            OcrError::input_limit(
                OcrStage::Recognition,
                InputLimitKind::TensorBytes,
                None,
                Some(isize::MAX as u64),
                "recognition tensor allocation exceeds platform limits",
            )
        })?;
    if width == 0 || width > u32::MAX as usize {
        return Err(OcrError::input_limit(
            OcrStage::Recognition,
            InputLimitKind::Dimensions,
            Some(width as u64),
            Some(u64::from(u32::MAX)),
            "recognition tensor dimensions exceed platform limits",
        ));
    }
    if bytes > isize::MAX as usize {
        return Err(OcrError::input_limit(
            OcrStage::Recognition,
            InputLimitKind::TensorBytes,
            Some(bytes as u64),
            Some(isize::MAX as u64),
            "recognition tensor dimensions exceed platform limits",
        ));
    }
    if let Some(max_width) = cfg.input_limits.max_width {
        if width > max_width as usize {
            return Err(OcrError::input_limit(OcrStage::Recognition, InputLimitKind::Width, Some(width as u64), Some(u64::from(max_width)), format!("recognition batch width {width} exceeds rec.input_limits.max_width={max_width}; tile long text lines, raise the limit, or explicitly use RecInputLimits::unrestricted()")));
        }
    }
    if let Some(max_pixels) = cfg.input_limits.max_batch_pixels {
        if pixels as u64 > max_pixels {
            return Err(OcrError::input_limit(OcrStage::Recognition, InputLimitKind::BatchPixelCount, Some(pixels as u64), Some(max_pixels), format!("recognition batch requests {pixels} padded pixels, exceeding rec.input_limits.max_batch_pixels={max_pixels}; reduce rec.batch_size or raise the limit")));
        }
    }
    Ok(())
}

fn resize_linear_opencv(
    img: &RgbImage,
    width: u32,
    height: u32,
    cancellation: &OcrCancellationToken,
) -> Result<RgbImage> {
    // Reimplement OpenCV-style bilinear resize for recognition preprocessing.
    // Small interpolation differences change OCR logits enough to matter in
    // parity fixtures, so this path avoids image crate sampling differences.
    let src_w = img.width();
    let src_h = img.height();
    let mut out = RgbImage::new(width, height);
    let scale_x = src_w as f32 / width.max(1) as f32;
    let scale_y = src_h as f32 / height.max(1) as f32;

    for y in 0..height {
        if y.is_multiple_of(16) {
            cancellation.checkpoint()?;
        }
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

    Ok(out)
}

fn linear_bounds(dst: u32, scale: f32, src_len: u32) -> (u32, u32, f32) {
    if src_len <= 1 {
        return (0, 0, 0.0);
    }

    // OpenCV maps destination pixel centers back to source pixel centers with a
    // half-pixel offset before choosing the two linear interpolation neighbors.
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
    let content = fs::read_to_string(path).map_err(|source| {
        let kind = if source.kind() == std::io::ErrorKind::NotFound {
            ModelErrorKind::NotFound
        } else {
            ModelErrorKind::Io
        };
        OcrError::model(
            kind,
            Some(path),
            format!("failed to read recognition dictionary {}", path.display()),
        )
        .with_source(source)
    })?;
    if content.trim().is_empty() {
        return Err(OcrError::model(
            ModelErrorKind::Incompatible,
            Some(path),
            format!("recognition dictionary {} is empty", path.display()),
        ));
    }
    let mut chars = Vec::new();
    chars.push("blank".to_string());
    chars.extend(content.lines().map(|line| line.trim_end().to_string()));
    chars.push(" ".to_string());
    Ok(chars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RecInputLimits;

    #[test]
    fn recognition_limits_reject_amplified_thin_inputs_before_allocation() {
        let cfg = RecConfig::new("rec.onnx", "dict.txt", [3, 48, 320], 6);
        let error = validate_batch_allocation(&cfg, 1, 48 * 200).unwrap_err();
        assert!(error
            .to_string()
            .contains("exceeds rec.input_limits.max_width=4096"));
    }

    #[test]
    fn recognition_limits_bound_total_batch_pixels() {
        let cfg = RecConfig::new("rec.onnx", "dict.txt", [3, 48, 320], 6)
            .with_input_limits(RecInputLimits::new(None, Some(48 * 320)));
        let error = validate_batch_allocation(&cfg, 2, 320).unwrap_err();
        assert!(error
            .to_string()
            .contains("exceeding rec.input_limits.max_batch_pixels"));
        assert!(matches!(
            error,
            OcrError::InputLimit {
                stage: OcrStage::Recognition,
                kind: InputLimitKind::BatchPixelCount,
                requested: Some(30720),
                limit: Some(15360),
                ..
            }
        ));
    }

    #[test]
    fn unrestricted_recognition_still_rejects_tensor_size_overflow() {
        let cfg = RecConfig::new("rec.onnx", "dict.txt", [3, 48, 320], 6)
            .with_input_limits(RecInputLimits::unrestricted());
        assert!(matches!(
            validate_batch_allocation(&cfg, usize::MAX, 320).unwrap_err(),
            OcrError::InputLimit {
                stage: OcrStage::Recognition,
                kind: InputLimitKind::Dimensions,
                requested: None,
                ..
            }
        ));
    }
}
