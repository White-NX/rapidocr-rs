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
    types::{AlignedToken, OcrTimings, RecAlignment, RecText},
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

            let mut valid_widths = self
                .cfg
                .return_tokens
                .then(|| Vec::with_capacity(chunk.len()));
            for (i, idx) in chunk.iter().enumerate() {
                cancellation.checkpoint()?;
                let img = &imgs[*idx];
                let (norm, valid_width) = self.resize_norm_img(img, batch_w, cancellation)?;
                batch.slice_mut(s![i, .., .., ..]).assign(&norm);
                if let Some(widths) = &mut valid_widths {
                    widths.push(valid_width);
                }
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
                let alignment_widths = valid_widths.as_ref().map(|widths| (batch_w, widths[i]));
                results[chunk[i]] = Some(
                    decode_ctc(
                        pred.slice(s![i, .., ..]),
                        &self.characters,
                        alignment_widths,
                        &self.cfg.dict_path,
                    )
                    .map_err(|error| error.with_model_path(&self.cfg.model_path))?,
                );
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
        img_w: usize,
        cancellation: &OcrCancellationToken,
    ) -> Result<(ndarray::Array3<f32>, usize)> {
        let [channels, img_h, _] = self.cfg.image_shape;
        if channels != 3 {
            return Err(OcrError::invalid_config(
                "only 3-channel recognition input is supported",
            ));
        }
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
        Ok((out, resized_w))
    }
}

fn decode_ctc(
    logits: ndarray::ArrayView2<'_, f32>,
    characters: &[String],
    alignment_widths: Option<(usize, usize)>,
    dict_path: &Path,
) -> Result<RecText> {
    let mut text = String::new();
    let mut confs = Vec::new();
    let mut last_idx = usize::MAX;
    let mut tokens = alignment_widths.map(|_| Vec::<AlignedToken>::new());

    for (time, timestep) in logits.outer_iter().enumerate() {
        let (idx, prob) = timestep
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, prob)| (idx, *prob))
            .unwrap_or((0, 0.0));
        if !prob.is_finite() {
            return Err(OcrError::incompatible_model(
                "recognition output contains a non-finite confidence",
            ));
        }

        // CTC decode: index 0 is blank and repeated non-blank labels collapse
        // into a single dictionary token, which can contain several code points.
        if idx == 0 || idx == last_idx {
            if idx != 0 {
                if let Some(token) = tokens.as_mut().and_then(|tokens| tokens.last_mut()) {
                    token.timestep_range.end = time + 1;
                }
            }
            last_idx = idx;
            continue;
        }

        let ch = characters.get(idx).ok_or_else(|| {
            OcrError::incompatible_model(format!(
                "recognition class {idx} is not present in the dictionary"
            ))
            .with_model_path(dict_path)
        })?;
        let text_start = text.len();
        text.push_str(ch);
        confs.push(prob);
        if let Some(tokens) = &mut tokens {
            tokens.push(AlignedToken {
                text_range: text_start..text.len(),
                score: prob,
                timestep_range: time..time + 1,
                x_center: 0.0,
                x_range: [0.0, 0.0],
            });
        }
        last_idx = idx;
    }

    let score = if confs.is_empty() {
        0.0
    } else {
        confs.iter().sum::<f32>() / confs.len() as f32
    };

    let alignment = match (alignment_widths, tokens) {
        (Some((input_width, valid_width)), Some(mut tokens)) => {
            let time_steps = logits.nrows();
            assign_selection_intervals(&mut tokens, time_steps, input_width, valid_width);
            Some(RecAlignment {
                input_width,
                valid_width,
                time_steps,
                tokens,
            })
        }
        _ => None,
    };

    Ok(RecText {
        text,
        score,
        alignment,
    })
}

fn assign_selection_intervals(
    tokens: &mut [AlignedToken],
    time_steps: usize,
    input_width: usize,
    valid_width: usize,
) {
    // One model timestep covers input_width / time_steps padded input pixels.
    // Divide by the actual resized crop width, not its ideal aspect ratio, so
    // short crops do not expand across the padding of a wider batch member.
    let scale = input_width as f64 / time_steps.max(1) as f64 / valid_width.max(1) as f64;
    for token in tokens.iter_mut() {
        let center = (token.timestep_range.start as f64 + token.timestep_range.end as f64) * 0.5;
        token.x_center = (center * scale).clamp(0.0, 1.0) as f32;
    }
    let mut start = 0.0;
    for index in 0..tokens.len() {
        let end = tokens
            .get(index + 1)
            .map_or(1.0, |next| (tokens[index].x_center + next.x_center) * 0.5);
        tokens[index].x_range = [start, end];
        start = end;
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

    fn ctc_prediction(sequence: &[(usize, f32)], classes: usize) -> ndarray::Array2<f32> {
        let mut prediction = ndarray::Array2::zeros((sequence.len(), classes));
        for (time, (label, score)) in sequence.iter().enumerate() {
            prediction[[time, *label]] = *score;
        }
        prediction
    }

    #[test]
    fn token_alignment_preserves_ctc_text_scores_and_utf8_ranges() {
        let characters = ["blank", "a", "🇨🇳", " ", "e\u{301}"].map(str::to_owned);
        let prediction = ctc_prediction(
            &[
                (0, 1.0),
                (1, 0.8),
                (1, 0.1),
                (0, 1.0),
                (1, 0.6),
                (2, 0.9),
                (2, 0.2),
                (3, 0.5),
                (4, 0.7),
                (0, 1.0),
            ],
            characters.len(),
        );
        let dict_path = Path::new("dictionary.txt");
        let plain = decode_ctc(prediction.view(), &characters, None, dict_path).unwrap();
        let aligned =
            decode_ctc(prediction.view(), &characters, Some((400, 400)), dict_path).unwrap();

        assert_eq!(plain.text, "aa🇨🇳 e\u{301}");
        assert_eq!(plain.text, aligned.text);
        assert_eq!(plain.score, aligned.score);
        assert!(plain.alignment.is_none());
        let alignment = aligned.alignment.unwrap();
        let tokens = alignment.tokens;
        assert_eq!(
            tokens
                .iter()
                .map(|token| token.text_range.clone())
                .collect::<Vec<_>>(),
            vec![0..1, 1..2, 2..10, 10..11, 11..14]
        );
        assert_eq!(
            tokens
                .iter()
                .map(|token| token.timestep_range.clone())
                .collect::<Vec<_>>(),
            vec![1..3, 4..5, 5..7, 7..8, 8..9]
        );
        assert_eq!(
            tokens.iter().map(|token| token.score).collect::<Vec<_>>(),
            vec![0.8, 0.6, 0.9, 0.5, 0.7]
        );
        assert_eq!(
            tokens
                .iter()
                .map(|token| &aligned.text[token.text_range.clone()])
                .collect::<String>(),
            aligned.text
        );
        assert_eq!(tokens[2].x_center, 0.6);
        assert_eq!(tokens[2].x_range, [0.525, 0.675]);
    }

    #[test]
    fn selection_intervals_use_real_valid_width_and_exclude_batch_padding() {
        let characters = ["blank", "a", "b"].map(str::to_owned);
        let make_prediction = |time_steps| {
            let mut sequence = vec![(0, 1.0); time_steps];
            sequence[10..13].fill((1, 0.9));
            sequence[20] = (2, 0.8);
            ctc_prediction(&sequence, characters.len())
        };
        let single = decode_ctc(
            make_prediction(100).view(),
            &characters,
            Some((400, 133)),
            Path::new("dict.txt"),
        )
        .unwrap();
        let padded = decode_ctc(
            make_prediction(200).view(),
            &characters,
            Some((800, 133)),
            Path::new("dict.txt"),
        )
        .unwrap();
        let single = single.alignment.unwrap();
        let padded = padded.alignment.unwrap();
        for (single, padded) in single.tokens.iter().zip(&padded.tokens) {
            assert_eq!(single.x_center, padded.x_center);
            assert_eq!(single.x_range, padded.x_range);
        }
        assert!((padded.tokens[0].x_center - 46.0 / 133.0).abs() < 1e-6);
        assert!((padded.tokens[0].x_range[1] - 64.0 / 133.0).abs() < 1e-6);

        let clipped = decode_ctc(
            make_prediction(200).view(),
            &characters,
            Some((800, 1)),
            Path::new("dict.txt"),
        )
        .unwrap();
        let clipped = clipped.alignment.unwrap();
        assert_eq!(clipped.tokens[0].x_range, [0.0, 1.0]);
        assert_eq!(clipped.tokens[1].x_range, [1.0, 1.0]);
        assert!(clipped.tokens.iter().all(|token| token.x_center == 1.0));
    }

    #[test]
    fn native_batch_alignment_tracks_resized_widths_after_reordering() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let cfg = RecConfig::new(
            root.join("paired-rec.onnx"),
            root.join("one-token.txt"),
            [3, 48, 320],
            2,
        )
        .with_return_tokens(true);
        let mut recognizer = TextRecognizer::new(cfg, InferenceOptions::default(), None).unwrap();
        let result = recognizer
            .recognize_timed(
                &[RgbImage::new(451, 49), RgbImage::new(101, 33)],
                &OcrCancellationToken::new(),
            )
            .unwrap();
        assert_eq!(result.texts[0].score, 0.2);
        assert_eq!(result.texts[1].score, 0.96);
        let wide = result.texts[0].alignment.as_ref().unwrap();
        let narrow = result.texts[1].alignment.as_ref().unwrap();
        assert_eq!(wide.input_width, 441);
        assert_eq!(wide.valid_width, 441);
        assert_eq!(narrow.input_width, 441);
        assert_eq!(narrow.valid_width, 147);
        assert_eq!(wide.tokens[0].x_center, 0.25);
        assert_eq!(narrow.tokens[0].x_center, 0.75);
    }

    #[test]
    fn missing_ctc_dictionary_entry_keeps_its_dictionary_error_path() {
        let prediction = ctc_prediction(&[(2, 0.9)], 3);
        let characters = ["blank", "a"].map(str::to_owned);
        for alignment in [None, Some((320, 100))] {
            let error = decode_ctc(
                prediction.view(),
                &characters,
                alignment,
                Path::new("dict.txt"),
            )
            .err()
            .unwrap()
            .with_model_path(Path::new("rec.onnx"));
            assert!(
                matches!(error, OcrError::Model { path: Some(ref path), .. } if path == Path::new("dict.txt"))
            );
        }
    }

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
