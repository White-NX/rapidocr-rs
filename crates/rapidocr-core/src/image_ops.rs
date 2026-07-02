use std::path::Path;

use anyhow::{bail, Context, Result};
use image::{
    imageops, metadata::Orientation, DynamicImage, ImageBuffer, ImageDecoder, ImageReader, Rgb,
    RgbImage, RgbaImage,
};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
use ndarray::Array4;

use crate::types::Quad;

pub fn load_rgb_image(path: impl AsRef<Path>) -> Result<RgbImage> {
    let path = path.as_ref();
    if !path.exists() {
        bail!("image file not found at {}", path.display());
    }
    if !path.is_file() {
        bail!("image path is not a file: {}", path.display());
    }
    let mut decoder = ImageReader::open(path)
        .with_context(|| format!("failed to open image {}", path.display()))?
        .into_decoder()
        .with_context(|| format!("failed to decode image {}", path.display()))?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut image = image::DynamicImage::from_decoder(decoder)
        .with_context(|| format!("failed to decode image {}", path.display()))?;
    image.apply_orientation(orientation);
    Ok(to_rgb_with_alpha_background(&image))
}

fn to_rgb_with_alpha_background(image: &DynamicImage) -> RgbImage {
    if !image.has_alpha() {
        return image.to_rgb8();
    }

    composite_alpha_to_contrast_background(&image.to_rgba8())
}

fn composite_alpha_to_contrast_background(image: &RgbaImage) -> RgbImage {
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

pub fn resize_image_within_bounds(
    img: &RgbImage,
    min_side_len: u32,
    max_side_len: u32,
) -> Result<(RgbImage, f32, f32)> {
    let (mut w, mut h) = img.dimensions();
    let original_w = w;
    let original_h = h;
    let mut current = img.clone();

    if w.max(h) > max_side_len {
        let ratio = max_side_len as f32 / w.max(h) as f32;
        w = round_to_multiple((w as f32 * ratio) as u32, 32).max(32);
        h = round_to_multiple((h as f32 * ratio) as u32, 32).max(32);
        current = imageops::resize(&current, w, h, imageops::FilterType::Triangle);
    }

    let (w2, h2) = current.dimensions();
    if w2.min(h2) < min_side_len {
        let ratio = min_side_len as f32 / w2.min(h2) as f32;
        w = round_to_multiple((w2 as f32 * ratio) as u32, 32).max(32);
        h = round_to_multiple((h2 as f32 * ratio) as u32, 32).max(32);
        current = imageops::resize(&current, w, h, imageops::FilterType::Triangle);
    }

    let (new_w, new_h) = current.dimensions();
    if new_w == 0 || new_h == 0 {
        bail!("resized image has zero dimension");
    }
    Ok((
        current,
        original_w as f32 / new_w as f32,
        original_h as f32 / new_h as f32,
    ))
}

pub fn apply_vertical_padding(
    img: &RgbImage,
    width_height_ratio: f32,
    min_height: u32,
) -> Result<(RgbImage, u32)> {
    let (w, h) = img.dimensions();
    let use_limit_ratio =
        width_height_ratio != -1.0 && w as f32 / h.max(1) as f32 > width_height_ratio;
    if h > min_height && !use_limit_ratio {
        return Ok((img.clone(), 0));
    }

    if width_height_ratio <= 0.0 && width_height_ratio != -1.0 {
        bail!("width_height_ratio must be positive or -1");
    }

    let base_h = if width_height_ratio == -1.0 {
        min_height
    } else {
        (w as f32 / width_height_ratio) as u32
    };
    let new_h = base_h.max(min_height) * 2;
    let padding_h = new_h.abs_diff(h) / 2;
    let mut out = ImageBuffer::from_pixel(w, h + padding_h * 2, Rgb([0, 0, 0]));
    imageops::replace(&mut out, img, 0, padding_h as i64);
    Ok((out, padding_h))
}

pub fn resize_to_multiple_for_det(
    img: &RgbImage,
    limit_side_len: u32,
    limit_min_side: bool,
) -> Result<RgbImage> {
    let (w, h) = img.dimensions();
    let ratio = if limit_min_side {
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

    let resize_w = round_to_multiple((w as f32 * ratio) as u32, 32).max(32);
    let resize_h = round_to_multiple((h as f32 * ratio) as u32, 32).max(32);
    Ok(imageops::resize(
        img,
        resize_w,
        resize_h,
        imageops::FilterType::Triangle,
    ))
}

pub fn rgb_to_nchw(img: &RgbImage, mean: [f32; 3], std: [f32; 3]) -> Array4<f32> {
    let (w, h) = img.dimensions();
    let mut array = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    for (x, y, pixel) in img.enumerate_pixels() {
        for c in 0..3 {
            array[[0, c, y as usize, x as usize]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    array
}

pub fn crop_axis_aligned(img: &RgbImage, bbox: &Quad) -> Result<RgbImage> {
    let (mut x0, mut y0, mut x1, mut y1) = bbox.axis_aligned_bounds();
    let (w, h) = img.dimensions();
    x0 = x0.min(w.saturating_sub(1));
    y0 = y0.min(h.saturating_sub(1));
    x1 = x1.min(w);
    y1 = y1.min(h);
    if x1 <= x0 || y1 <= y0 {
        bail!("invalid crop bounds");
    }
    Ok(imageops::crop_imm(img, x0, y0, x1 - x0, y1 - y0).to_image())
}

pub fn crop_perspective(img: &RgbImage, bbox: &Quad) -> Result<RgbImage> {
    let mut bbox = bbox.clone().ordered();
    const REPLICATE_PAD: u32 = 2;
    if is_near_image_edge(&bbox, img.width(), img.height(), REPLICATE_PAD) {
        let padded = replicate_border(img, REPLICATE_PAD);
        for point in &mut bbox.points {
            point[0] += REPLICATE_PAD as f32;
            point[1] += REPLICATE_PAD as f32;
        }
        return crop_perspective_ordered(&padded, &bbox);
    }

    crop_perspective_ordered(img, &bbox)
}

fn crop_perspective_ordered(img: &RgbImage, bbox: &Quad) -> Result<RgbImage> {
    let crop_w = bbox.crop_width();
    let crop_h = bbox.crop_height();
    if crop_w == 0 || crop_h == 0 {
        bail!("invalid perspective crop size");
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
        return crop_axis_aligned(img, &bbox);
    };

    let mut out = ImageBuffer::from_pixel(crop_w, crop_h, Rgb([0, 0, 0]));
    warp_into(
        img,
        &projection,
        Interpolation::Bicubic,
        Rgb([0, 0, 0]),
        &mut out,
    );

    if out.height() as f32 / out.width().max(1) as f32 >= 1.5 {
        Ok(imageops::rotate270(&out))
    } else {
        Ok(out)
    }
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

pub fn round_to_multiple(value: u32, divisor: u32) -> u32 {
    ((value + divisor / 2) / divisor) * divisor
}

#[cfg(test)]
mod tests {
    use image::{Rgba, RgbaImage};

    use super::*;

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
}
