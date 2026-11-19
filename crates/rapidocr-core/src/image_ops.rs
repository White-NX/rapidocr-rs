use std::{borrow::Cow, path::Path};

use image::{
    imageops, metadata::Orientation, DynamicImage, ImageBuffer, ImageDecoder, ImageReader, Rgb,
    RgbImage, RgbaImage,
};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
use ndarray::Array4;

use crate::{
    cancellation::OcrCancellationToken,
    config::{DetInputLimits, DetOverflowBehavior},
    error::{InputLimitKind, OcrError, OcrResult as Result, OcrStage},
    types::{OcrTextMapping, Quad},
};

pub(crate) fn load_rgb_image(path: impl AsRef<Path>) -> Result<RgbImage> {
    let path = path.as_ref();
    let metadata = std::fs::metadata(path).map_err(|source| {
        let message = if source.kind() == std::io::ErrorKind::NotFound {
            format!("image file not found at {}", path.display())
        } else {
            format!("failed to inspect image {}", path.display())
        };
        OcrError::invalid_input(Some(path), message).with_source(source)
    })?;
    if !metadata.is_file() {
        return Err(OcrError::invalid_input(
            Some(path),
            format!("image path is not a file: {}", path.display()),
        ));
    }
    let mut decoder = ImageReader::open(path)
        .map_err(|source| {
            OcrError::invalid_input(
                Some(path),
                format!("failed to open image {}", path.display()),
            )
            .with_source(source)
        })?
        .into_decoder()
        .map_err(|source| image_decode_error(path, source))?;
    // Match ImageReader::decode's allocation policy even though we keep the
    // decoder to read EXIF orientation before constructing the DynamicImage.
    let decoded_bytes = decoder.total_bytes();
    let mut decode_limits = image::Limits::default();
    let allocation_limit = decode_limits.max_alloc;
    decode_limits.reserve(decoded_bytes).map_err(|source| {
        OcrError::input_limit(
            OcrStage::Preprocessing,
            InputLimitKind::ImageBytes,
            Some(decoded_bytes),
            allocation_limit,
            format!("decoded image {} requires {decoded_bytes} bytes, exceeding the image decoder allocation limit", path.display()),
        ).with_source(source)
    })?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut image = image::DynamicImage::from_decoder(decoder)
        .map_err(|source| image_decode_error(path, source))?;
    image.apply_orientation(orientation);
    Ok(to_rgb_with_alpha_background(&image))
}

fn image_decode_error(path: &Path, source: image::ImageError) -> OcrError {
    let message = format!("failed to decode image {}", path.display());
    let error = if let image::ImageError::Limits(limits) = &source {
        let kind = match limits.kind() {
            image::error::LimitErrorKind::InsufficientMemory => InputLimitKind::ImageBytes,
            _ => InputLimitKind::Dimensions,
        };
        OcrError::input_limit(OcrStage::Preprocessing, kind, None, None, message)
    } else {
        OcrError::invalid_input(Some(path), message)
    };
    error.with_source(source)
}

fn padding_dimension_error(message: &str) -> OcrError {
    OcrError::input_limit(
        OcrStage::Preprocessing,
        InputLimitKind::Dimensions,
        None,
        Some(u64::from(u32::MAX)),
        message,
    )
}

fn to_rgb_with_alpha_background(image: &DynamicImage) -> RgbImage {
    if !image.has_alpha() {
        return image.to_rgb8();
    }

    composite_alpha_to_contrast_background(&image.to_rgba8())
}

fn composite_alpha_to_contrast_background(image: &RgbaImage) -> RgbImage {
    // Transparent OCR inputs often contain either dark text or light text with no
    // explicit background. Choose the opposite background by foreground luminance
    // so text remains visible after alpha is flattened to RGB.
    let mut luminance_sum = 0.0f32;
    let mut non_transparent_count = 0usize;
    for pixel in image.pixels() {
        if pixel[3] == 0 {
            continue;
        }
        luminance_sum +=
            0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
        non_transparent_count += 1;
    }

    let bg = if non_transparent_count == 0 || luminance_sum / (non_transparent_count as f32) < 128.0
    {
        [255.0, 255.0, 255.0]
    } else {
        [0.0, 0.0, 0.0]
    };

    let mut out = RgbImage::new(image.width(), image.height());
    for (x, y, pixel) in image.enumerate_pixels() {
        let alpha = pixel[3] as f32 / 255.0;
        out.put_pixel(
            x,
            y,
            Rgb([
                (pixel[0] as f32 * alpha + bg[0] * (1.0 - alpha)) as u8,
                (pixel[1] as f32 * alpha + bg[1] * (1.0 - alpha)) as u8,
                (pixel[2] as f32 * alpha + bg[2] * (1.0 - alpha)) as u8,
            ]),
        );
    }
    out
}

pub(crate) fn resize_image_within_bounds(
    img: &RgbImage,
    min_side_len: u32,
    max_side_len: u32,
) -> Result<(Cow<'_, RgbImage>, f32, f32)> {
    let (mut w, mut h) = img.dimensions();
    let original_w = w;
    let original_h = h;
    let mut current = Cow::Borrowed(img);

    // The outer pipeline keeps dimensions aligned to 32 so later detector
    // preprocessing does not introduce a second large geometry shift.
    if w.max(h) > max_side_len {
        let ratio = max_side_len as f32 / w.max(h) as f32;
        w = round_to_multiple((w as f32 * ratio) as u32, 32).max(32);
        h = round_to_multiple((h as f32 * ratio) as u32, 32).max(32);
        current = Cow::Owned(imageops::resize(
            current.as_ref(),
            w,
            h,
            imageops::FilterType::Triangle,
        ));
    }

    let (w2, h2) = current.dimensions();
    if w2.min(h2) < min_side_len {
        let ratio = min_side_len as f32 / w2.min(h2) as f32;
        w = round_to_multiple((w2 as f32 * ratio) as u32, 32).max(32);
        h = round_to_multiple((h2 as f32 * ratio) as u32, 32).max(32);
        current = Cow::Owned(imageops::resize(
            current.as_ref(),
            w,
            h,
            imageops::FilterType::Triangle,
        ));
    }

    let (new_w, new_h) = current.dimensions();
    if new_w == 0 || new_h == 0 {
        return Err(OcrError::invalid_input(
            None,
            "resized image has zero dimension",
        ));
    }
    Ok((
        current,
        original_w as f32 / new_w as f32,
        original_h as f32 / new_h as f32,
    ))
}

pub(crate) fn apply_vertical_padding(
    img: &RgbImage,
    width_height_ratio: f32,
    min_height: u32,
) -> Result<(Cow<'_, RgbImage>, u32)> {
    let (w, h) = img.dimensions();
    let use_limit_ratio =
        width_height_ratio != -1.0 && w as f32 / h.max(1) as f32 > width_height_ratio;
    if h > min_height && !use_limit_ratio {
        return Ok((Cow::Borrowed(img), 0));
    }

    if width_height_ratio <= 0.0 && width_height_ratio != -1.0 {
        return Err(OcrError::invalid_config(
            "width_height_ratio must be positive or -1",
        ));
    }

    // RapidOCR pads very short or very wide images before detection to avoid
    // detector misses on thin text lines. The caller subtracts this top padding
    // before mapping boxes back to original image coordinates.
    let base_h = if width_height_ratio == -1.0 {
        min_height
    } else {
        (w as f32 / width_height_ratio) as u32
    };
    let new_h = base_h.max(min_height).checked_mul(2).ok_or_else(|| {
        padding_dimension_error("vertical padding height exceeds image dimension limits")
    })?;
    let padding_h = new_h.abs_diff(h) / 2;
    let padded_h = h
        .checked_add(padding_h.checked_mul(2).ok_or_else(|| {
            padding_dimension_error("vertical padding height exceeds image dimension limits")
        })?)
        .ok_or_else(|| {
            padding_dimension_error("padded image height exceeds image dimension limits")
        })?;
    let mut out = ImageBuffer::from_pixel(w, padded_h, Rgb([0, 0, 0]));
    imageops::replace(&mut out, img, 0, padding_h as i64);
    Ok((Cow::Owned(out), padding_h))
}

pub(crate) fn resize_to_multiple_for_det(
    img: &RgbImage,
    limit_side_len: u32,
    limit_min_side: bool,
    input_limits: &DetInputLimits,
) -> Result<RgbImage> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(OcrError::invalid_input(
            None,
            "detector input image must have non-zero dimensions",
        ));
    }
    let requested_ratio = if limit_min_side {
        if w.min(h) < limit_side_len {
            limit_side_len as f32 / w.min(h) as f32
        } else {
            1.0
        }
    } else if w.max(h) > limit_side_len {
        limit_side_len as f32 / w.max(h) as f32
    } else {
        1.0
    };
    let requested_dimensions = aligned_dimensions(w, h, requested_ratio, false);
    let exceeds_limits = !dimensions_within_limits(requested_dimensions, input_limits);

    let (resize_w, resize_h) = if !exceeds_limits
        || input_limits.overflow_behavior == DetOverflowBehavior::Allow
    {
        requested_dimensions
    } else if input_limits.overflow_behavior == DetOverflowBehavior::Reject {
        let (requested_w, requested_h) = requested_dimensions;
        let (kind, requested, limit) = if let Some(max_side) = input_limits
            .max_side_len
            .filter(|limit| requested_w.max(requested_h) > *limit)
        {
            (
                InputLimitKind::SideLength,
                u64::from(requested_w.max(requested_h)),
                Some(u64::from(max_side)),
            )
        } else {
            (
                InputLimitKind::PixelCount,
                u64::from(requested_w) * u64::from(requested_h),
                input_limits.max_pixels,
            )
        };
        return Err(OcrError::input_limit(OcrStage::Detection, kind, Some(requested), limit, format!(
            "detector resize requested {requested_w}x{requested_h} pixels, exceeding configured input limits (max_side_len={}, max_pixels={}); use downscale, raise the limits, or explicitly select allow",
            display_optional_limit(input_limits.max_side_len.map(u64::from)),
            display_optional_limit(input_limits.max_pixels),
        )));
    } else {
        fit_dimensions_within_limits(w, h, requested_ratio, input_limits)
    };
    Ok(imageops::resize(
        img,
        resize_w,
        resize_h,
        imageops::FilterType::Triangle,
    ))
}

fn fit_dimensions_within_limits(
    w: u32,
    h: u32,
    requested_ratio: f32,
    limits: &DetInputLimits,
) -> (u32, u32) {
    // Alignment and the mandatory 32-pixel minimum make a purely analytical
    // scale insufficient for very thin inputs. Binary search the largest
    // aligned shape that satisfies both independent limits.
    let mut low = 0.0_f32;
    let mut high = requested_ratio;
    let mut best = (32, 32);
    for _ in 0..48 {
        let mid = low + (high - low) / 2.0;
        let dimensions = aligned_dimensions(w, h, mid, true);
        if dimensions_within_limits(dimensions, limits) {
            best = dimensions;
            low = mid;
        } else {
            high = mid;
        }
    }
    best
}

fn aligned_dimensions(w: u32, h: u32, ratio: f32, round_down: bool) -> (u32, u32) {
    let align = if round_down {
        floor_to_multiple
    } else {
        round_to_multiple
    };
    (
        align((w as f32 * ratio) as u32, 32).max(32),
        align((h as f32 * ratio) as u32, 32).max(32),
    )
}

fn dimensions_within_limits((w, h): (u32, u32), limits: &DetInputLimits) -> bool {
    limits
        .max_side_len
        .is_none_or(|max_side_len| w.max(h) <= max_side_len)
        && limits
            .max_pixels
            .is_none_or(|max_pixels| u64::from(w) * u64::from(h) <= max_pixels)
}

fn display_optional_limit(limit: Option<u64>) -> String {
    limit.map_or_else(|| "unlimited".to_owned(), |value| value.to_string())
}

pub(crate) fn rgb_to_nchw(
    img: &RgbImage,
    mean: [f32; 3],
    std: [f32; 3],
    cancellation: &OcrCancellationToken,
) -> Result<Array4<f32>> {
    let (w, h) = img.dimensions();
    let mut array = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    for (x, y, pixel) in img.enumerate_pixels() {
        if x == 0 && y.is_multiple_of(32) {
            cancellation.checkpoint()?;
        }
        for c in 0..3 {
            array[[0, c, y as usize, x as usize]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    Ok(array)
}

fn crop_axis_aligned(
    img: &RgbImage,
    bbox: &Quad,
    return_transform: bool,
) -> Result<(RgbImage, Option<CropTransform>)> {
    let (mut x0, mut y0, mut x1, mut y1) = bbox.axis_aligned_bounds();
    let (w, h) = img.dimensions();
    x0 = x0.min(w.saturating_sub(1));
    y0 = y0.min(h.saturating_sub(1));
    x1 = x1.min(w);
    y1 = y1.min(h);
    if x1 <= x0 || y1 <= y0 {
        return Err(OcrError::invalid_input(None, "invalid crop bounds"));
    }
    let crop = imageops::crop_imm(img, x0, y0, x1 - x0, y1 - y0).to_image();
    let transform = return_transform.then(|| CropTransform {
        to_image: Projection::translate(x0 as f32, y0 as f32),
        matrix_to_image: [
            [1.0, 0.0, x0 as f32],
            [0.0, 1.0, y0 as f32],
            [0.0, 0.0, 1.0],
        ],
        crop_size: crop.dimensions(),
        image_size: img.dimensions(),
    });
    Ok((crop, transform))
}

pub(crate) fn crop_perspective(
    img: &RgbImage,
    bbox: &Quad,
    return_transform: bool,
) -> Result<(RgbImage, Option<CropTransform>)> {
    let mut bbox = bbox.clone().ordered();
    const REPLICATE_PAD: u32 = 2;
    // OpenCV perspective warps near image edges sample replicated border pixels.
    // Add a small replicated border to match that behavior for tiny edge text.
    if is_near_image_edge(&bbox, img.width(), img.height(), REPLICATE_PAD) {
        let padded = replicate_border(img, REPLICATE_PAD);
        for point in &mut bbox.points {
            point[0] += REPLICATE_PAD as f32;
            point[1] += REPLICATE_PAD as f32;
        }
        let (crop, mut transform) = crop_perspective_ordered(&padded, &bbox, return_transform)?;
        if let Some(transform) = &mut transform {
            transform.to_image =
                Projection::translate(-(REPLICATE_PAD as f32), -(REPLICATE_PAD as f32))
                    * transform.to_image;
            transform.matrix_to_image = multiply_matrix(
                [
                    [1.0, 0.0, -(REPLICATE_PAD as f32)],
                    [0.0, 1.0, -(REPLICATE_PAD as f32)],
                    [0.0, 0.0, 1.0],
                ],
                transform.matrix_to_image,
            );
            transform.image_size = img.dimensions();
        }
        return Ok((crop, transform));
    }

    crop_perspective_ordered(img, &bbox, return_transform)
}

fn crop_perspective_ordered(
    img: &RgbImage,
    bbox: &Quad,
    return_transform: bool,
) -> Result<(RgbImage, Option<CropTransform>)> {
    let crop_w = bbox.crop_width();
    let crop_h = bbox.crop_height();
    if crop_w == 0 || crop_h == 0 {
        return Err(OcrError::invalid_input(
            None,
            "invalid perspective crop size",
        ));
    }

    let from = [
        (bbox.points[0][0], bbox.points[0][1]),
        (bbox.points[1][0], bbox.points[1][1]),
        (bbox.points[2][0], bbox.points[2][1]),
        (bbox.points[3][0], bbox.points[3][1]),
    ];
    let to = [
        (0.0, 0.0),
        (crop_w as f32, 0.0),
        (crop_w as f32, crop_h as f32),
        (0.0, crop_h as f32),
    ];

    let Some(projection) = Projection::from_control_points(from, to) else {
        return crop_axis_aligned(img, bbox, return_transform);
    };

    let mut out = ImageBuffer::from_pixel(crop_w, crop_h, Rgb([0, 0, 0]));
    warp_into(
        img,
        &projection,
        Interpolation::Bicubic,
        Rgb([0, 0, 0]),
        &mut out,
    );

    // RapidOCR rotates tall text crops before recognition so recognizer input is
    // predominantly horizontal.
    let quarter_turn = out.height() as f32 / out.width().max(1) as f32 >= 1.5;
    let out = if quarter_turn {
        imageops::rotate270(&out)
    } else {
        out
    };
    let transform = if return_transform {
        let mut to_image = projection.invert();
        // The warp library keeps its matrix private. Record the equivalent
        // inverse homography from the very same unmodified control points,
        // before any source-image clipping, rather than inferring it later
        // from a line/selection quad. The image warp itself stays unchanged.
        let mut matrix_to_image = inverse_crop_matrix(bbox, crop_w, crop_h).ok_or_else(|| {
            OcrError::invalid_input(None, "invalid perspective selection transform")
        })?;
        if quarter_turn {
            // Exact inverse of imageops::rotate270 on pixel-center coordinates:
            // (x, y) in the rotated crop samples (crop_w - 1 - y, x).
            let inverse_rotation = Projection::from_matrix([
                0.0,
                -1.0,
                crop_w.saturating_sub(1) as f32,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ])
            .expect("quarter-turn matrix is invertible");
            to_image = to_image * inverse_rotation;
            matrix_to_image = multiply_matrix(
                matrix_to_image,
                [
                    [0.0, -1.0, crop_w.saturating_sub(1) as f32],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            );
        }
        Some(CropTransform {
            to_image,
            matrix_to_image,
            crop_size: out.dimensions(),
            image_size: img.dimensions(),
        })
    } else {
        None
    };
    Ok((out, transform))
}

/// Actual recognition-crop pixel-center mapping, retained only for token output.
///
/// This is saved while making the crop, before source line boxes are clipped.
/// It includes edge replication and fallback crops as well as perspective warps.
pub(crate) struct CropTransform {
    to_image: Projection,
    matrix_to_image: [[f32; 3]; 3],
    crop_size: (u32, u32),
    image_size: (u32, u32),
}

impl CropTransform {
    pub(crate) fn identity(width: u32, height: u32) -> Self {
        Self {
            to_image: Projection::translate(0.0, 0.0),
            matrix_to_image: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            crop_size: (width, height),
            image_size: (width, height),
        }
    }

    pub(crate) fn map_pipeline_to_source(
        &mut self,
        ratio_w: f32,
        ratio_h: f32,
        padding_top: u32,
        original_size: (u32, u32),
    ) {
        self.to_image = Projection::scale(ratio_w, ratio_h)
            * Projection::translate(0.0, -(padding_top as f32))
            * self.to_image;
        self.matrix_to_image = multiply_matrix(
            [
                [ratio_w, 0.0, 0.0],
                [0.0, ratio_h, -(padding_top as f32) * ratio_h],
                [0.0, 0.0, 1.0],
            ],
            self.matrix_to_image,
        );
        self.image_size = original_size;
    }

    pub(crate) fn selection_mapping(&self, applied_degrees: u16) -> OcrTextMapping {
        let max_x = self.crop_size.0.saturating_sub(1) as f32;
        let max_y = self.crop_size.1.saturating_sub(1) as f32;
        let normalized_to_crop = if applied_degrees == 180 {
            [[-max_x, 0.0, max_x], [0.0, -max_y, max_y], [0.0, 0.0, 1.0]]
        } else {
            [[max_x, 0.0, 0.0], [0.0, max_y, 0.0], [0.0, 0.0, 1.0]]
        };
        OcrTextMapping::new(
            multiply_matrix(self.matrix_to_image, normalized_to_crop),
            [self.image_size.0, self.image_size.1],
        )
    }

    pub(crate) fn selection_quad(&self, x_range: [f32; 2], applied_degrees: u16) -> Quad {
        // Normalize onto the crop's sampled pixel centers, matching public Quad
        // bounds (0..width-1). Do not rebuild a homography from the clipped line box.
        let max_x = self.crop_size.0.saturating_sub(1) as f32;
        let max_y = self.crop_size.1.saturating_sub(1) as f32;
        let mut quad = Quad::from_xyxy(x_range[0] * max_x, 0.0, x_range[1] * max_x, max_y);
        for point in &mut quad.points {
            if applied_degrees == 180 {
                point[0] = max_x - point[0];
                point[1] = max_y - point[1];
            }
            let (x, y) = self.to_image * (point[0], point[1]);
            *point = [x, y];
        }
        quad.clip(self.image_size.0, self.image_size.1);
        quad
    }
}

fn multiply_matrix(left: [[f32; 3]; 3], right: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|col| (0..3).map(|k| left[row][k] * right[k][col]).sum())
    })
}

/// Inverse of the rectangular crop's homography, using its exact control points.
fn inverse_crop_matrix(bbox: &Quad, width: u32, height: u32) -> Option<[[f32; 3]; 3]> {
    let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = bbox.points.map(|point| point.map(f64::from));
    let dx1 = x1 - x2;
    let dx2 = x3 - x2;
    let dx3 = x0 - x1 + x2 - x3;
    let dy1 = y1 - y2;
    let dy2 = y3 - y2;
    let dy3 = y0 - y1 + y2 - y3;
    let (g, h) = if dx3 == 0.0 && dy3 == 0.0 {
        (0.0, 0.0)
    } else {
        let determinant = dx1 * dy2 - dx2 * dy1;
        if determinant == 0.0 {
            return None;
        }
        (
            (dx3 * dy2 - dx2 * dy3) / determinant,
            (dx1 * dy3 - dx3 * dy1) / determinant,
        )
    };
    let matrix = [
        [
            (x1 - x0 + g * x1) / f64::from(width),
            (x3 - x0 + h * x3) / f64::from(height),
            x0,
        ],
        [
            (y1 - y0 + g * y1) / f64::from(width),
            (y3 - y0 + h * y3) / f64::from(height),
            y0,
        ],
        [g / f64::from(width), h / f64::from(height), 1.0],
    ]
    .map(|row| row.map(|value| value as f32));
    matrix
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(matrix)
}

fn is_near_image_edge(bbox: &Quad, width: u32, height: u32, pad: u32) -> bool {
    let max_x = width.saturating_sub(1) as f32;
    let max_y = height.saturating_sub(1) as f32;
    let pad = pad as f32;
    bbox.points.iter().any(|point| {
        point[0] < pad || point[1] < pad || point[0] > max_x - pad || point[1] > max_y - pad
    })
}

fn replicate_border(img: &RgbImage, pad: u32) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut out = RgbImage::new(width + pad * 2, height + pad * 2);
    for y in 0..out.height() {
        for x in 0..out.width() {
            let src_x = x.saturating_sub(pad).min(width.saturating_sub(1));
            let src_y = y.saturating_sub(pad).min(height.saturating_sub(1));
            out.put_pixel(x, y, *img.get_pixel(src_x, src_y));
        }
    }
    out
}

pub(crate) fn round_to_multiple(value: u32, divisor: u32) -> u32 {
    ((value.saturating_add(divisor / 2)) / divisor) * divisor
}

fn floor_to_multiple(value: u32, divisor: u32) -> u32 {
    (value / divisor) * divisor
}

#[cfg(test)]
mod tests {
    use image::{Rgba, RgbaImage};

    use super::*;

    fn assert_points(actual: &Quad, expected: [[f32; 2]; 4]) {
        for (actual, expected) in actual.points.iter().zip(expected) {
            assert!(
                (actual[0] - expected[0]).abs() < 1e-3,
                "x: {actual:?} != {expected:?}"
            );
            assert!(
                (actual[1] - expected[1]).abs() < 1e-3,
                "y: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn selection_transform_preserves_perspective_edge_padding_and_source_clipping() {
        let image = RgbImage::from_fn(96, 48, |x, y| Rgb([x as u8, y as u8, 0]));
        let bbox = Quad::new([[1.0, 2.0], [61.0, 2.0], [41.0, 32.0], [1.0, 32.0]]);
        let (plain, disabled) = crop_perspective(&image, &bbox, false).unwrap();
        let (crop, transform) = crop_perspective(&image, &bbox, true).unwrap();
        assert_eq!(plain, crop);
        assert!(disabled.is_none());
        assert_eq!(crop.dimensions(), (60, 36));
        let mut transform = transform.unwrap();
        // Analytic inverse for this trapezoid: x=1+60u/(1+v/2),
        // y=2+45v/(1+v/2), with u=x_crop/60 and v=y_crop/36.
        let bottom_denominator = 1.0 + (35.0 / 36.0) * 0.5;
        let expected = [
            [15.75, 2.0],
            [45.25, 2.0],
            [
                1.0 + 44.25 / bottom_denominator,
                2.0 + 43.75 / bottom_denominator,
            ],
            [
                1.0 + 14.75 / bottom_denominator,
                2.0 + 43.75 / bottom_denominator,
            ],
        ];
        assert_points(&transform.selection_quad([0.25, 0.75], 0), expected);
        let mapping = transform.selection_mapping(0);
        assert_eq!(mapping.source_size, [96, 48]);
        assert_points(&mapping.selection_quad([0.25, 0.75]).unwrap(), expected);

        transform.map_pipeline_to_source(3.0, 0.5, 10, (100, 12));
        let expected = expected.map(|[x, y]| {
            [
                (x * 3.0).clamp(0.0, 99.0),
                ((y - 10.0) * 0.5).clamp(0.0, 11.0),
            ]
        });
        assert_points(&transform.selection_quad([0.25, 0.75], 0), expected);
        let mapping = transform.selection_mapping(0);
        assert_eq!(mapping.source_size, [100, 12]);
        let unclipped = mapping.project_point([0.75, 0.0]).unwrap();
        assert!((unclipped[0] - 135.75).abs() < 1e-3);
        assert!((unclipped[1] + 4.0).abs() < 1e-3);
        assert_points(&mapping.selection_quad([0.25, 0.75]).unwrap(), expected);
    }

    #[test]
    fn tall_crop_selection_inverts_quarter_turn_and_classifier_rotation() {
        let image = RgbImage::new(64, 128);
        let bbox = Quad::from_xyxy(10.0, 20.0, 30.0, 100.0);
        let (crop, transform) = crop_perspective(&image, &bbox, true).unwrap();
        assert_eq!(crop.dimensions(), (80, 20));
        let transform = transform.unwrap();
        assert_points(
            &transform.selection_quad([0.25, 0.5], 0),
            [[29.0, 39.75], [29.0, 59.5], [10.0, 59.5], [10.0, 39.75]],
        );
        assert_points(
            &transform.selection_quad([0.25, 0.5], 180),
            [[10.0, 79.25], [10.0, 59.5], [29.0, 59.5], [29.0, 79.25]],
        );
        for degrees in [0, 180] {
            let mapping = transform.selection_mapping(degrees);
            assert_eq!(mapping.source_size, [64, 128]);
            assert_points(
                &mapping.selection_quad([0.25, 0.5]).unwrap(),
                transform.selection_quad([0.25, 0.5], degrees).points,
            );
        }
    }

    #[test]
    fn degenerate_perspective_keeps_the_actual_axis_aligned_fallback_mapping() {
        let image = RgbImage::new(64, 32);
        let bbox = Quad::new([[3.0, 4.0], [23.0, 8.0], [33.0, 10.0], [13.0, 6.0]]);
        let (crop, transform) = crop_perspective(&image, &bbox, true).unwrap();
        assert_eq!(crop.dimensions(), (30, 6));
        let transform = transform.unwrap();
        assert_points(
            &transform.selection_quad([0.25, 0.5], 0),
            [[10.25, 4.0], [17.5, 4.0], [17.5, 9.0], [10.25, 9.0]],
        );
        assert_points(
            &transform
                .selection_mapping(0)
                .selection_quad([0.25, 0.5])
                .unwrap(),
            [[10.25, 4.0], [17.5, 4.0], [17.5, 9.0], [10.25, 9.0]],
        );
    }

    #[test]
    fn normalized_source_mapping_matches_the_unclipped_warp_at_interior_points() {
        let image = RgbImage::new(4096, 2160);
        let boxes = [
            Quad::new([[2.0, 1.0], [1720.0, 15.0], [1630.0, 320.0], [7.0, 410.0]]),
            Quad::new([
                [1200.125, 600.25],
                [3600.5, 550.75],
                [3220.0, 1100.0],
                [1430.0, 1200.0],
            ]),
            Quad::new([
                [2000.0, 800.0],
                [2130.0, 850.0],
                [1800.0, 1900.0],
                [1690.0, 1810.0],
            ]),
        ];
        for bbox in boxes {
            let (_, transform) = crop_perspective(&image, &bbox, true).unwrap();
            let mut transform = transform.unwrap();
            transform.map_pipeline_to_source(0.75, 1.25, 32, (3072, 2620));
            let max_x = transform.crop_size.0.saturating_sub(1) as f32;
            let max_y = transform.crop_size.1.saturating_sub(1) as f32;
            for degrees in [0, 180] {
                let mapping = transform.selection_mapping(degrees);
                for u in [0.0, 0.13, 0.5, 0.87, 1.0] {
                    for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                        let (x, y) = if degrees == 180 {
                            ((1.0 - u) * max_x, (1.0 - v) * max_y)
                        } else {
                            (u * max_x, v * max_y)
                        };
                        let expected = transform.to_image * (x, y);
                        let actual = mapping.project_point([u, v]).unwrap();
                        assert!(
                            (actual[0] - expected.0).abs() < 0.005
                                && (actual[1] - expected.1).abs() < 0.005,
                            "mapped {actual:?}, warp {expected:?}, normalized [{u}, {v}], rotation {degrees}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn one_pixel_crop_dimensions_allow_singular_selection_matrices() {
        for (width, height) in [(1, 32), (48, 1), (1, 1)] {
            let transform = CropTransform::identity(width, height);
            for degrees in [0, 180] {
                let mapping = transform.selection_mapping(degrees);
                assert_eq!(mapping.source_size, [width, height]);
                assert_points(
                    &mapping.selection_quad([0.2, 0.8]).unwrap(),
                    transform.selection_quad([0.2, 0.8], degrees).points,
                );
                assert!(mapping.project_point([0.5, 0.5]).is_some());
            }
        }
    }

    #[test]
    fn unchanged_pipeline_images_are_borrowed() {
        let image = RgbImage::new(320, 48);
        let (resized, ratio_w, ratio_h) = resize_image_within_bounds(&image, 30, 2000).unwrap();
        assert!(matches!(resized, Cow::Borrowed(_)));
        assert_eq!((ratio_w, ratio_h), (1.0, 1.0));
        let (padded, padding_top) = apply_vertical_padding(&resized, 8.0, 30).unwrap();
        assert!(matches!(padded, Cow::Borrowed(_)));
        assert_eq!(padding_top, 0);
    }

    #[test]
    fn excessive_padding_height_returns_an_error_before_allocation() {
        let image = RgbImage::new(32, 1);
        let error = apply_vertical_padding(&image, -1.0, u32::MAX).unwrap_err();
        assert!(error
            .to_string()
            .contains("vertical padding height exceeds image dimension limits"));
    }

    #[test]
    fn alpha_images_are_composited_onto_contrast_background() {
        let mut black_text = RgbaImage::from_pixel(2, 1, Rgba([0, 0, 0, 0]));
        black_text.put_pixel(0, 0, Rgba([0, 0, 0, 255]));

        let black_out = composite_alpha_to_contrast_background(&black_text);

        assert_eq!(black_out.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(black_out.get_pixel(1, 0).0, [255, 255, 255]);

        let mut white_text = RgbaImage::from_pixel(2, 1, Rgba([0, 0, 0, 0]));
        white_text.put_pixel(0, 0, Rgba([255, 255, 255, 255]));

        let white_out = composite_alpha_to_contrast_background(&white_text);

        assert_eq!(white_out.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(white_out.get_pixel(1, 0).0, [0, 0, 0]);
    }

    #[test]
    fn replicate_border_extends_edge_pixels() {
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([1, 2, 3]));
        img.put_pixel(1, 0, Rgb([4, 5, 6]));
        img.put_pixel(0, 1, Rgb([7, 8, 9]));
        img.put_pixel(1, 1, Rgb([10, 11, 12]));

        let padded = replicate_border(&img, 1);

        assert_eq!(padded.get_pixel(0, 0).0, [1, 2, 3]);
        assert_eq!(padded.get_pixel(3, 0).0, [4, 5, 6]);
        assert_eq!(padded.get_pixel(0, 3).0, [7, 8, 9]);
        assert_eq!(padded.get_pixel(3, 3).0, [10, 11, 12]);
        assert_eq!(padded.get_pixel(2, 2).0, [10, 11, 12]);
    }

    #[test]
    fn detector_min_side_resize_keeps_pathological_aspect_ratios_bounded() {
        let image = RgbImage::new(32, 1984);

        let resized =
            resize_to_multiple_for_det(&image, 736, true, &DetInputLimits::default()).unwrap();

        assert!(resized.width().min(resized.height()) < 736);
        assert!(resized.width().max(resized.height()) <= 4096);
        assert_eq!(resized.width() % 32, 0);
        assert_eq!(resized.height() % 32, 0);
    }

    #[test]
    fn detector_min_side_resize_preserves_normal_minimum_side_behavior() {
        let image = RgbImage::new(640, 480);

        let resized =
            resize_to_multiple_for_det(&image, 736, true, &DetInputLimits::default()).unwrap();

        assert_eq!(resized.dimensions(), (992, 736));
    }

    #[test]
    fn detector_resize_rejects_limit_overflow_before_allocating_target() {
        let image = RgbImage::new(640, 480);
        for (limits, expected_kind, expected_request, expected_limit) in [
            (
                DetInputLimits::new(Some(512), None, DetOverflowBehavior::Reject),
                InputLimitKind::SideLength,
                992,
                512,
            ),
            (
                DetInputLimits::new(None, Some(1024), DetOverflowBehavior::Reject),
                InputLimitKind::PixelCount,
                992 * 736,
                1024,
            ),
        ] {
            let error = resize_to_multiple_for_det(&image, 736, true, &limits).unwrap_err();
            assert!(error.to_string().contains("requested 992x736"));
            assert!(
                matches!(error, OcrError::InputLimit { stage: OcrStage::Detection, kind, requested: Some(requested), limit: Some(limit), .. }
                if kind == expected_kind && requested == expected_request && limit == expected_limit)
            );
        }
    }

    #[test]
    fn detector_resize_allow_explicitly_ignores_limits() {
        let image = RgbImage::new(640, 480);
        let limits = DetInputLimits::new(Some(512), Some(32 * 32), DetOverflowBehavior::Allow);

        let resized = resize_to_multiple_for_det(&image, 736, true, &limits).unwrap();

        assert_eq!(resized.dimensions(), (992, 736));
    }

    #[test]
    fn detector_resize_enforces_pixel_only_limit_on_thin_input() {
        let image = RgbImage::new(1, 200);
        let limits = DetInputLimits::new(None, Some(32 * 320), DetOverflowBehavior::Downscale);

        let resized = resize_to_multiple_for_det(&image, 736, true, &limits).unwrap();

        assert!(u64::from(resized.width()) * u64::from(resized.height()) <= 32 * 320);
        assert_eq!(resized.width(), 32);
        assert_eq!(resized.height() % 32, 0);
    }
}
