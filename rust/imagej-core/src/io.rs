//! Minimal image I/O using the `image` crate.
//!
//! Mirrors `ij.io.ImageReader` behavior where we map image formats
//! to the appropriate processor types. AWT-dependent GUI features are omitted.

use crate::processor::{ByteProcessor, ColorProcessor, FloatProcessor, ShortProcessor};

/// Supported image types after loading. Mirrors ImagePlus type system.
#[derive(Debug, Clone)]
pub enum ImageData {
    Byte(ByteProcessor),
    Short(ShortProcessor),
    Float(FloatProcessor),
    Color(ColorProcessor),
}

/// Loads an image from a file path using the `image` crate.
///
/// Returns the appropriate processor based on bit depth:
/// - 8-bit grayscale → ByteProcessor
/// - 16-bit grayscale → ShortProcessor
/// - 32-bit float → FloatProcessor (if supported)
/// - RGB/RGBA → ColorProcessor
pub fn load_image(path: &str) -> Result<ImageData, String> {
    let img = image::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;

    match img {
        image::DynamicImage::ImageLuma8(buf) => {
            let w = buf.width() as usize;
            let h = buf.height() as usize;
            let pixels: Vec<u8> = buf.pixels().map(|p| p.0[0]).collect();
            Ok(ImageData::Byte(ByteProcessor::from_pixels(w, h, pixels)))
        }
        image::DynamicImage::ImageLumaA8(buf) => {
            // 8-bit grayscale + alpha -> extract luma
            let w = buf.width() as usize;
            let h = buf.height() as usize;
            let pixels: Vec<u8> = buf.pixels().map(|p| p.0[0]).collect();
            Ok(ImageData::Byte(ByteProcessor::from_pixels(w, h, pixels)))
        }
        image::DynamicImage::ImageRgb8(buf) => {
            let w = buf.width() as usize;
            let h = buf.height() as usize;
            let pixels: Vec<u32> = buf
                .pixels()
                .map(|p| {
                    let [r, g, b] = p.0;
                    0xff000000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
                })
                .collect();
            Ok(ImageData::Color(ColorProcessor::from_pixels(w, h, pixels)))
        }
        image::DynamicImage::ImageRgba8(buf) => {
            let w = buf.width() as usize;
            let h = buf.height() as usize;
            let pixels: Vec<u32> = buf
                .pixels()
                .map(|p| {
                    let [r, g, b, a] = p.0;
                    // image crate uses RGBA order, we store as ARGB
                    (a as u32) << 24 | (r as u32) << 16 | (g as u32) << 8 | (b as u32)
                })
                .collect();
            Ok(ImageData::Color(ColorProcessor::from_pixels(w, h, pixels)))
        }
        image::DynamicImage::ImageLuma16(buf) => {
            let w = buf.width() as usize;
            let h = buf.height() as usize;
            let pixels: Vec<u16> = buf.pixels().map(|p| p.0[0]).collect();
            Ok(ImageData::Short(ShortProcessor::from_pixels(w, h, pixels)))
        }
        image::DynamicImage::ImageLumaA16(buf) => {
            let w = buf.width() as usize;
            let h = buf.height() as usize;
            let pixels: Vec<u16> = buf.pixels().map(|p| p.0[0]).collect();
            Ok(ImageData::Short(ShortProcessor::from_pixels(w, h, pixels)))
        }
        image::DynamicImage::ImageRgb16(buf) => {
            // 16-bit RGB -> convert to 8-bit then to ColorProcessor
            let img = image::DynamicImage::ImageRgb16(buf).into_rgb8();
            let w = img.width() as usize;
            let h = img.height() as usize;
            let pixels: Vec<u32> = img
                .pixels()
                .map(|p| {
                    let [r, g, b] = p.0;
                    0xff000000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
                })
                .collect();
            Ok(ImageData::Color(ColorProcessor::from_pixels(w, h, pixels)))
        }
        image::DynamicImage::ImageRgba16(buf) => {
            let img = image::DynamicImage::ImageRgba16(buf).into_rgba8();
            let w = img.width() as usize;
            let h = img.height() as usize;
            let pixels: Vec<u32> = img
                .pixels()
                .map(|p| {
                    let [r, g, b, a] = p.0;
                    (a as u32) << 24 | (r as u32) << 16 | (g as u32) << 8 | (b as u32)
                })
                .collect();
            Ok(ImageData::Color(ColorProcessor::from_pixels(w, h, pixels)))
        }
        image::DynamicImage::ImageRgb32F(buf) => {
            let buf = image::DynamicImage::ImageRgb32F(buf).to_luma32f();
            let w = buf.width() as usize;
            let h = buf.height() as usize;
            let pixels: Vec<f32> = buf.pixels().map(|p| p.0[0]).collect();
            Ok(ImageData::Float(FloatProcessor::from_pixels(w, h, pixels)))
        }
        image::DynamicImage::ImageRgba32F(buf) => {
            let buf = image::DynamicImage::ImageRgba32F(buf).to_luma32f();
            let w = buf.width() as usize;
            let h = buf.height() as usize;
            let pixels: Vec<f32> = buf.pixels().map(|p| p.0[0]).collect();
            Ok(ImageData::Float(FloatProcessor::from_pixels(w, h, pixels)))
        }
        _ => Err(format!(
            "Unsupported image format or bit depth for {}",
            path
        )),
    }
}

/// Saves an image to a file. Format is inferred from the extension.
pub fn save_image(path: &str, img: &ImageData) -> Result<(), String> {
    let dyn_img = match img {
        ImageData::Byte(bp) => {
            let (width, height) = checked_dimensions(bp.width, bp.height)?;
            image::DynamicImage::ImageLuma8(
                image::ImageBuffer::from_raw(width, height, bp.pixels.clone()).ok_or_else(
                    || "ByteProcessor pixel length does not match dimensions".to_string(),
                )?,
            )
        }
        ImageData::Short(sp) => {
            let (width, height) = checked_dimensions(sp.width, sp.height)?;
            image::DynamicImage::ImageLuma16(
                image::ImageBuffer::from_raw(width, height, sp.pixels.clone()).ok_or_else(
                    || "ShortProcessor pixel length does not match dimensions".to_string(),
                )?,
            )
        }
        ImageData::Color(cp) => {
            let (width, height) = checked_dimensions(cp.width, cp.height)?;
            // Build Rgba<u8> buffer from ARGB
            let raw: Vec<u8> = cp
                .pixels
                .iter()
                .flat_map(|&p| {
                    [
                        ((p >> 16) & 0xff) as u8,
                        ((p >> 8) & 0xff) as u8,
                        (p & 0xff) as u8,
                        ((p >> 24) & 0xff) as u8,
                    ]
                })
                .collect();
            image::DynamicImage::ImageRgba8(
                image::ImageBuffer::from_raw(width, height, raw).ok_or_else(|| {
                    "ColorProcessor pixel length does not match dimensions".to_string()
                })?,
            )
        }
        ImageData::Float(_) => return Err("Float image save not yet implemented".to_string()),
    };

    let path = std::path::Path::new(path);
    dyn_img
        .save(path)
        .map_err(|e| format!("Failed to save {}: {}", path.display(), e))
}

fn checked_dimensions(width: usize, height: usize) -> Result<(u32, u32), String> {
    let width = u32::try_from(width).map_err(|_| "image width exceeds u32::MAX".to_string())?;
    let height = u32::try_from(height).map_err(|_| "image height exceeds u32::MAX".to_string())?;
    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_file(extension: &str) -> PathBuf {
        let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "imagej-core-{}-{}.{extension}",
            std::process::id(),
            id
        ))
    }

    #[test]
    fn byte_png_round_trip() {
        let path = temp_file("png");
        let source = ImageData::Byte(ByteProcessor::from_pixels(2, 2, vec![0, 64, 128, 255]));

        save_image(path.to_str().unwrap(), &source).unwrap();
        let loaded = load_image(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);

        match loaded {
            ImageData::Byte(bp) => {
                assert_eq!((bp.width, bp.height), (2, 2));
                assert_eq!(bp.pixels, vec![0, 64, 128, 255]);
            }
            _ => panic!("expected ByteProcessor"),
        }
    }

    #[test]
    fn color_png_preserves_argb() {
        let path = temp_file("png");
        let source = ImageData::Color(ColorProcessor::from_pixels(
            2,
            1,
            vec![0x00112233, 0xffaabbcc],
        ));

        save_image(path.to_str().unwrap(), &source).unwrap();
        let loaded = load_image(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);

        match loaded {
            ImageData::Color(cp) => assert_eq!(cp.pixels, vec![0x00112233, 0xffaabbcc]),
            _ => panic!("expected ColorProcessor"),
        }
    }

    #[test]
    fn invalid_pixel_length_returns_error() {
        let path = temp_file("png");
        let mut bp = ByteProcessor::new(2, 2);
        bp.pixels.pop();

        let error = save_image(path.to_str().unwrap(), &ImageData::Byte(bp)).unwrap_err();

        assert!(error.contains("pixel length"));
        assert!(!path.exists());
    }

    #[test]
    fn float_save_returns_error() {
        let path = temp_file("png");
        let source = ImageData::Float(FloatProcessor::new(1, 1));

        let error = save_image(path.to_str().unwrap(), &source).unwrap_err();

        assert!(error.contains("not yet implemented"));
        assert!(!path.exists());
    }
}
