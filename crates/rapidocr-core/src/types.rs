//! Public OCR output and timing data types.

use std::ops::Range;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
/// Four-point text box.
///
/// Points are stored as `[x, y]` pairs. Public OCR output uses image pixel
/// coordinates after clipping to the source image bounds.
pub struct Quad {
    /// Box corners as four `[x, y]` points.
    pub points: [[f32; 2]; 4],
}

impl Quad {
    /// Creates a quadrilateral from four `[x, y]` points.
    pub const fn new(points: [[f32; 2]; 4]) -> Self {
        Self { points }
    }

    /// Creates an axis-aligned quadrilateral from inclusive corner coordinates.
    pub fn from_xyxy(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            points: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        }
    }

    pub(crate) fn scale(&mut self, ratio_w: f32, ratio_h: f32) {
        for point in &mut self.points {
            point[0] *= ratio_w;
            point[1] *= ratio_h;
        }
    }

    pub(crate) fn clip(&mut self, width: u32, height: u32) {
        let max_x = width.saturating_sub(1) as f32;
        let max_y = height.saturating_sub(1) as f32;
        for point in &mut self.points {
            point[0] = point[0].clamp(0.0, max_x);
            point[1] = point[1].clamp(0.0, max_y);
        }
    }

    pub(crate) fn axis_aligned_bounds(&self) -> (u32, u32, u32, u32) {
        let min_x = self
            .points
            .iter()
            .map(|p| p[0])
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0) as u32;
        let min_y = self
            .points
            .iter()
            .map(|p| p[1])
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0) as u32;
        let max_x = self
            .points
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .max(0.0) as u32;
        let max_y = self
            .points
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .max(0.0) as u32;
        (min_x, min_y, max_x, max_y)
    }

    pub(crate) fn ordered(mut self) -> Self {
        self.points = order_like_get_mini_boxes(self.points);
        self
    }

    pub(crate) fn order_clockwise_in_place(&mut self) {
        self.points = order_points_clockwise(self.points);
    }

    pub(crate) fn crop_width(&self) -> u32 {
        self.width_f32().floor().max(1.0) as u32
    }

    pub(crate) fn crop_height(&self) -> u32 {
        self.height_f32().floor().max(1.0) as u32
    }

    pub(crate) fn width_f32(&self) -> f32 {
        let top = distance(self.points[0], self.points[1]);
        let bottom = distance(self.points[3], self.points[2]);
        top.max(bottom)
    }

    pub(crate) fn height_f32(&self) -> f32 {
        let left = distance(self.points[0], self.points[3]);
        let right = distance(self.points[1], self.points[2]);
        left.max(right)
    }

    pub(crate) fn short_side(&self) -> f32 {
        self.width_f32().min(self.height_f32())
    }

    pub(crate) fn contains_point(&self, x: f32, y: f32) -> bool {
        let mut inside = false;
        let points = &self.points;
        let mut j = points.len() - 1;
        for i in 0..points.len() {
            let xi = points[i][0];
            let yi = points[i][1];
            let xj = points[j][0];
            let yj = points[j][1];
            if ((yi > y) != (yj > y))
                && (x < (xj - xi) * (y - yi) / ((yj - yi) + f32::EPSILON) + xi)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

fn order_like_get_mini_boxes(mut points: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    points.sort_by(|a, b| a[0].total_cmp(&b[0]));

    let (index_1, index_4) = if points[1][1] > points[0][1] {
        (0, 1)
    } else {
        (1, 0)
    };
    let (index_2, index_3) = if points[3][1] > points[2][1] {
        (2, 3)
    } else {
        (3, 2)
    };

    [
        points[index_1],
        points[index_2],
        points[index_3],
        points[index_4],
    ]
}

fn order_points_clockwise(mut points: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    points.sort_by(|a, b| a[0].total_cmp(&b[0]));

    let mut left_most = [points[0], points[1]];
    let mut right_most = [points[2], points[3]];
    left_most.sort_by(|a, b| a[1].total_cmp(&b[1]));
    right_most.sort_by(|a, b| a[1].total_cmp(&b[1]));

    [left_most[0], right_most[0], right_most[1], left_most[1]]
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

pub(crate) struct RecText {
    pub text: String,
    pub score: f32,
    pub alignment: Option<RecAlignment>,
}

pub(crate) struct RecAlignment {
    pub input_width: usize,
    pub valid_width: usize,
    pub time_steps: usize,
    pub tokens: Vec<AlignedToken>,
}

pub(crate) struct AlignedToken {
    pub text_range: Range<usize>,
    pub score: f32,
    pub timestep_range: Range<usize>,
    pub x_center: f32,
    pub x_range: [f32; 2],
}

/// One dictionary token and its estimated selection cell.
///
/// A token may contain several Unicode code points (for example a flag emoji),
/// or be a space. It is not necessarily one character or grapheme. Selection
/// cells come from greedy CTC alignment and cover the crop's full height; they
/// are not pixel-precise glyph detection boxes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OcrToken {
    /// Half-open UTF-8 byte range in [`OcrLine::text`].
    ///
    /// Ranges are contiguous in decode order and reconstruct the complete text.
    /// These offsets are not Unicode code-point or JavaScript UTF-16 indices.
    pub text_range: Range<usize>,
    /// Confidence at the first retained CTC timestep, as used in the line score.
    ///
    /// This is recognition confidence, not a measure of coordinate accuracy.
    pub score: f32,
    /// Half-open run of consecutive equal, non-blank greedy CTC labels.
    ///
    /// Blank timesteps are excluded. Equal labels separated by blank produce
    /// separate tokens. Indices refer to [`OcrTextAlignment::time_steps`].
    pub timestep_range: Range<usize>,
    /// Center of the CTC run, normalized to the valid recognition crop width.
    ///
    /// Uses the horizontal, post-rotation recognition orientation, clipped to
    /// `[0, 1]`. Batch padding is excluded from the coordinate scale.
    pub x_center: f32,
    /// Estimated selection interval in the same normalized coordinates.
    ///
    /// Interior boundaries are midpoints between adjacent run centers. The
    /// first/last cells extend to the crop edges; spaces retain their own cells.
    /// Tokens emitted in padded input may have zero-width cells at the edge.
    pub x_range: [f32; 2],
    /// Estimated selection quadrilateral in source-image pixel coordinates.
    ///
    /// Corners follow `(x_start, top)`, `(x_end, top)`, `(x_end, bottom)`,
    /// `(x_start, bottom)` in recognition orientation, mapped through the actual
    /// crop, rotations, resize and padding, then clipped to the source image.
    /// Corner order therefore need not start at the source-image top left.
    /// `run_path` coordinates use the image after EXIF orientation, like line boxes.
    pub bbox: Quad,
}

impl OcrToken {
    /// Creates a token with explicit text, alignment, and selection coordinates.
    pub fn new(
        text_range: Range<usize>,
        score: f32,
        timestep_range: Range<usize>,
        x_center: f32,
        x_range: [f32; 2],
        bbox: Quad,
    ) -> Self {
        Self {
            text_range,
            score,
            timestep_range,
            x_center,
            x_range,
            bbox,
        }
    }
}

/// Maps the normalized recognition rectangle to unclipped source-image pixels.
///
/// The map already includes crop orientation, classifier rotation, pixel-center
/// scaling, perspective/fallback cropping, edge replication, and detector
/// pipeline resize/padding. Store this once per line to reconstruct selection
/// cells from their one-dimensional intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OcrTextMapping {
    /// Row-major 3x3 projective matrix multiplying the column vector `[u, v, 1]`.
    ///
    /// `u` and `v` range across `[0, 1]` in the post-rotation recognition crop.
    /// Divide the first two resulting components by the third to obtain source
    /// pixel coordinates. Those coordinates are intentionally not clipped.
    pub matrix: [[f32; 3]; 3],
    /// Source image `[width, height]`, used to clip reconstructed selections.
    ///
    /// For `run_path`, these dimensions follow EXIF correction, like line boxes.
    pub source_size: [u32; 2],
}

impl OcrTextMapping {
    /// Creates a mapping with explicit matrix coefficients and source dimensions.
    pub const fn new(matrix: [[f32; 3]; 3], source_size: [u32; 2]) -> Self {
        Self {
            matrix,
            source_size,
        }
    }

    /// Projects a normalized point without clipping it to the source image.
    ///
    /// Returns `None` for non-finite inputs/results or a zero homogeneous divisor.
    /// Singular matrices for one-pixel-wide/high crops are valid if projection
    /// still has a finite nonzero divisor; no matrix inversion is required.
    pub fn project_point(&self, [u, v]: [f32; 2]) -> Option<[f32; 2]> {
        if !u.is_finite() || !v.is_finite() {
            return None;
        }
        let m = &self.matrix;
        let divisor = m[2][0] * u + m[2][1] * v + m[2][2];
        if !divisor.is_finite() || divisor == 0.0 {
            return None;
        }
        let x = (m[0][0] * u + m[0][1] * v + m[0][2]) / divisor;
        let y = (m[1][0] * u + m[1][1] * v + m[1][2]) / divisor;
        (x.is_finite() && y.is_finite()).then_some([x, y])
    }

    /// Reconstructs a full-height selection cell and clips its source coordinates.
    ///
    /// Corners preserve recognition order. Returns `None` for invalid mappings,
    /// empty source dimensions, or an unordered/non-normalized interval.
    pub fn selection_quad(&self, x_range: [f32; 2]) -> Option<Quad> {
        if self.source_size.contains(&0)
            || x_range[0] > x_range[1]
            || !x_range.iter().all(|x| (0.0..=1.0).contains(x))
        {
            return None;
        }
        let mut points = [
            [x_range[0], 0.0],
            [x_range[1], 0.0],
            [x_range[1], 1.0],
            [x_range[0], 1.0],
        ];
        for point in &mut points {
            *point = self.project_point(*point)?;
        }
        let mut quad = Quad::new(points);
        quad.clip(self.source_size[0], self.source_size[1]);
        Some(quad)
    }
}

/// Optional CTC alignment produced by [`crate::config::RecConfig::return_tokens`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OcrTextAlignment {
    /// Padded ONNX input width shared by the recognition batch, in pixels.
    pub input_width: usize,
    /// Actual resized width occupied by this crop before right-side padding.
    pub valid_width: usize,
    /// Number of output timesteps in the model's `[N, T, C]` tensor.
    pub time_steps: usize,
    /// Tokens in CTC decode order, including whitespace and multi-code-point entries.
    pub tokens: Vec<OcrToken>,
    /// Unclipped normalized-crop mapping for compact selection persistence.
    ///
    /// The OCR pipeline supplies this whenever token output is enabled. `None`
    /// supports older serialized alignment and manually constructed metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapping: Option<OcrTextMapping>,
}

impl OcrTextAlignment {
    /// Creates alignment metadata without exposing the full probability tensor.
    pub fn new(
        input_width: usize,
        valid_width: usize,
        time_steps: usize,
        tokens: Vec<OcrToken>,
    ) -> Self {
        Self {
            input_width,
            valid_width,
            time_steps,
            tokens,
            mapping: None,
        }
    }

    /// Attaches the unclipped source mapping while preserving existing constructors.
    pub fn with_mapping(mut self, mapping: OcrTextMapping) -> Self {
        self.mapping = Some(mapping);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
/// Direction classification and the rotation applied to a recognition crop.
///
/// This describes the classifier's additional rotation after perspective
/// cropping. It does not include EXIF orientation or the automatic quarter-turn
/// used when cropping tall detected boxes, and does not change the output box.
pub struct OcrRotation {
    /// Winning classifier label, commonly `"0"` or `"180"`.
    pub label: String,
    /// Confidence of the winning classifier label, even below the rotation threshold.
    pub score: f32,
    /// Additional clockwise rotation actually applied to the crop, in degrees.
    ///
    /// The pipeline reports `180` when the label contains `"180"` and its score
    /// strictly exceeds [`crate::config::ClsConfig::thresh`]; otherwise it reports `0`.
    pub applied_degrees: u16,
}

impl OcrRotation {
    /// Creates direction-classification metadata with the applied rotation.
    pub fn new(label: impl Into<String>, score: f32, applied_degrees: u16) -> Self {
        Self {
            label: label.into(),
            score,
            applied_degrees,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
/// One OCR result line.
pub struct OcrLine {
    /// Detected or synthetic text box.
    pub bbox: Quad,
    /// Recognized text. Detection-only output leaves this empty.
    pub text: String,
    /// Mean recognition confidence. Detection-only output uses `0.0`.
    pub score: f32,
    /// DB postprocessing confidence for the detected box, before box expansion.
    ///
    /// Available in detection-only and detection-plus-recognition output.
    /// `None` means detection was disabled or no detection metadata was supplied.
    #[serde(default)]
    pub detection_score: Option<f32>,
    /// Direction classification and the rotation applied before recognition.
    ///
    /// `None` means classification was disabled or no rotation metadata was supplied.
    #[serde(default)]
    pub rotation: Option<OcrRotation>,
    /// Optional CTC tokens and estimated selection cells.
    ///
    /// `None` when token output or recognition is disabled. Absent metadata is
    /// omitted from JSON so enabling this capability does not change default output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_alignment: Option<OcrTextAlignment>,
}

impl OcrLine {
    /// Creates one recognized OCR line.
    pub fn new(bbox: Quad, text: impl Into<String>, score: f32) -> Self {
        Self {
            bbox,
            text: text.into(),
            score,
            detection_score: None,
            rotation: None,
            text_alignment: None,
        }
    }

    /// Creates one detection-only OCR line with empty text and zero recognition score.
    ///
    /// Use [`OcrLine::with_detection_score`] to attach a detector confidence.
    pub fn detection_only(bbox: Quad) -> Self {
        Self::new(bbox, String::new(), 0.0)
    }

    /// Attaches the detector confidence without changing the recognition score.
    pub fn with_detection_score(mut self, score: f32) -> Self {
        self.detection_score = Some(score);
        self
    }

    /// Attaches direction classification and applied rotation metadata.
    pub fn with_rotation(mut self, rotation: OcrRotation) -> Self {
        self.rotation = Some(rotation);
        self
    }

    /// Attaches CTC tokens and estimated selection cells.
    pub fn with_text_alignment(mut self, alignment: OcrTextAlignment) -> Self {
        self.text_alignment = Some(alignment);
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
/// OCR output lines in reading order.
pub struct OcrOutput {
    /// Recognized lines.
    pub lines: Vec<OcrLine>,
}

impl OcrOutput {
    /// Creates an OCR output from lines in reading order.
    pub fn new(lines: Vec<OcrLine>) -> Self {
        Self { lines }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
/// Millisecond timings for OCR stages.
pub struct OcrTimings {
    /// Image decoding, EXIF orientation, and RGB conversion time.
    pub image_load_ms: f64,
    /// High-level pipeline resize and padding time before detection.
    pub pipeline_preprocess_ms: f64,
    /// Detection input resize and tensor normalization time.
    pub det_preprocess_ms: f64,
    /// Detection ONNX inference time.
    pub det_inference_ms: f64,
    /// Detection DB postprocessing time.
    pub det_postprocess_ms: f64,
    /// Perspective crop generation time.
    pub crop_ms: f64,
    /// Classification input resize and tensor normalization time.
    pub cls_preprocess_ms: f64,
    /// Classification ONNX inference time.
    pub cls_inference_ms: f64,
    /// Classification decode and rotation time.
    pub cls_postprocess_ms: f64,
    /// Recognition input resize and tensor normalization time.
    pub rec_preprocess_ms: f64,
    /// Recognition ONNX inference time.
    pub rec_inference_ms: f64,
    /// Recognition CTC decode time.
    pub rec_decode_ms: f64,
    /// Empty-text and low-score output filtering time.
    pub output_filter_ms: f64,
    /// Total elapsed time for the requested run.
    pub total_ms: f64,
}

impl OcrTimings {
    /// Adds another timing sample into this one field by field.
    pub fn add_assign(&mut self, other: &Self) {
        self.image_load_ms += other.image_load_ms;
        self.pipeline_preprocess_ms += other.pipeline_preprocess_ms;
        self.det_preprocess_ms += other.det_preprocess_ms;
        self.det_inference_ms += other.det_inference_ms;
        self.det_postprocess_ms += other.det_postprocess_ms;
        self.crop_ms += other.crop_ms;
        self.cls_preprocess_ms += other.cls_preprocess_ms;
        self.cls_inference_ms += other.cls_inference_ms;
        self.cls_postprocess_ms += other.cls_postprocess_ms;
        self.rec_preprocess_ms += other.rec_preprocess_ms;
        self.rec_inference_ms += other.rec_inference_ms;
        self.rec_decode_ms += other.rec_decode_ms;
        self.output_filter_ms += other.output_filter_ms;
        self.total_ms += other.total_ms;
    }

    /// Divides every timing field by `denominator`.
    ///
    /// A zero denominator returns the input unchanged.
    pub fn div_by(self, denominator: f64) -> Self {
        if denominator == 0.0 {
            return self;
        }

        Self {
            image_load_ms: self.image_load_ms / denominator,
            pipeline_preprocess_ms: self.pipeline_preprocess_ms / denominator,
            det_preprocess_ms: self.det_preprocess_ms / denominator,
            det_inference_ms: self.det_inference_ms / denominator,
            det_postprocess_ms: self.det_postprocess_ms / denominator,
            crop_ms: self.crop_ms / denominator,
            cls_preprocess_ms: self.cls_preprocess_ms / denominator,
            cls_inference_ms: self.cls_inference_ms / denominator,
            cls_postprocess_ms: self.cls_postprocess_ms / denominator,
            rec_preprocess_ms: self.rec_preprocess_ms / denominator,
            rec_inference_ms: self.rec_inference_ms / denominator,
            rec_decode_ms: self.rec_decode_ms / denominator,
            output_filter_ms: self.output_filter_ms / denominator,
            total_ms: self.total_ms / denominator,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
/// OCR output paired with per-stage timings.
pub struct TimedOcrOutput {
    /// Recognized OCR lines.
    pub output: OcrOutput,
    /// Timing breakdown in milliseconds.
    pub timings: OcrTimings,
}

impl TimedOcrOutput {
    /// Creates an OCR output paired with its timing breakdown.
    pub fn new(output: OcrOutput, timings: OcrTimings) -> Self {
        Self { output, timings }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_mini_boxes_order_matches_python_left_group_logic() {
        let points = [[10.0, 7.0], [2.0, 1.0], [12.0, 2.0], [0.0, 6.0]];
        let ordered = order_like_get_mini_boxes(points);
        assert_eq!(ordered, [[2.0, 1.0], [12.0, 2.0], [10.0, 7.0], [0.0, 6.0]]);
    }

    #[test]
    fn order_points_clockwise_matches_python_filter_logic() {
        let points = [[10.0, 7.0], [2.0, 1.0], [12.0, 2.0], [0.0, 6.0]];
        let ordered = order_points_clockwise(points);
        assert_eq!(ordered, [[2.0, 1.0], [12.0, 2.0], [10.0, 7.0], [0.0, 6.0]]);
    }

    #[test]
    fn order_points_clockwise_handles_axis_aligned_shuffle() {
        let points = [[10.0, 10.0], [0.0, 0.0], [0.0, 10.0], [10.0, 0.0]];
        let ordered = order_points_clockwise(points);
        assert_eq!(
            ordered,
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        );
    }
}
