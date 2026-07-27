//! Port of `ij.process.ImageConverter` — converts an `ImagePlus` between
//! pixel types (8-bit, 16-bit, 32-bit float, RGB).
//!
//! Mirrors the core `convertToGray8/Gray16/Gray32/RGB` behavior, operating
//! directly on `ImagePlus`. Calibration-aware scaling and recording are omitted
//! for the headless port (those depend on GUI/Prefs state).

use crate::ij_core::ImagePlus;
use crate::io::ImageData;
use crate::processor::{ByteProcessor, ColorProcessor, FloatProcessor, ShortProcessor};

/// Converts an `ImagePlus` to a different pixel type in place.
pub struct ImageConverter<'a> {
    imp: &'a mut ImagePlus,
}

impl<'a> ImageConverter<'a> {
    /// Creates a converter for the given image.
    pub fn new(imp: &'a mut ImagePlus) -> Self {
        ImageConverter { imp }
    }

    /// Converts to 8-bit grayscale. Errors if the image is a composite/stack
    /// that cannot be represented (unsupported conversion).
    pub fn convert_to_gray8(&mut self) {
        let proc = match self.imp.get_processor().cloned() {
            Some(p) => p,
            None => return,
        };
        let converted = match proc {
            ImageData::Byte(_) => return, // already 8-bit
            ImageData::Short(s) => ImageData::Byte(s.to_byte(true)),
            ImageData::Float(f) => ImageData::Byte(f.to_byte(true)),
            ImageData::Color(c) => ImageData::Byte(c.to_byte()),
        };
        self.imp.set_processor(converted);
    }

    /// Converts to 16-bit grayscale.
    pub fn convert_to_gray16(&mut self) {
        let proc = match self.imp.get_processor().cloned() {
            Some(p) => p,
            None => return,
        };
        let converted = match proc {
            ImageData::Short(_) => return, // already 16-bit
            ImageData::Byte(b) => ImageData::Short(b.to_short()),
            ImageData::Float(f) => ImageData::Short(f.to_short(true)),
            ImageData::Color(c) => ImageData::Short(c.to_short()),
        };
        self.imp.set_processor(converted);
    }

    /// Converts to 32-bit float grayscale.
    pub fn convert_to_gray32(&mut self) {
        let proc = match self.imp.get_processor().cloned() {
            Some(p) => p,
            None => return,
        };
        let converted = match proc {
            ImageData::Float(_) => return, // already 32-bit
            ImageData::Byte(b) => ImageData::Float(b.to_float()),
            ImageData::Short(s) => ImageData::Float(s.to_float()),
            ImageData::Color(c) => ImageData::Float(c.to_float()),
        };
        self.imp.set_processor(converted);
    }

    /// Converts to 24-bit RGB.
    pub fn convert_to_rgb(&mut self) {
        let proc = match self.imp.get_processor().cloned() {
            Some(p) => p,
            None => return,
        };
        let converted = match proc {
            ImageData::Color(_) => return, // already RGB
            ImageData::Byte(b) => ImageData::Color(b.to_rgb()),
            ImageData::Short(s) => ImageData::Color(s.to_rgb()),
            ImageData::Float(f) => ImageData::Color(f.to_rgb()),
        };
        self.imp.set_processor(converted);
    }
}

/// Convenience: convert a single `ImageData` to 8-bit (without an `ImagePlus`).
pub fn to_gray8(proc: &ImageData) -> ByteProcessor {
    match proc {
        ImageData::Byte(b) => b.clone(),
        ImageData::Short(s) => s.to_byte(true),
        ImageData::Float(f) => f.to_byte(true),
        ImageData::Color(c) => c.to_byte(),
    }
}

/// Convenience: convert a single `ImageData` to 16-bit.
pub fn to_gray16(proc: &ImageData) -> ShortProcessor {
    match proc {
        ImageData::Short(s) => s.clone(),
        ImageData::Byte(b) => b.to_short(),
        ImageData::Float(f) => f.to_short(true),
        ImageData::Color(c) => c.to_short(),
    }
}

/// Convenience: convert a single `ImageData` to 32-bit float.
pub fn to_gray32(proc: &ImageData) -> FloatProcessor {
    match proc {
        ImageData::Float(f) => f.clone(),
        ImageData::Byte(b) => b.to_float(),
        ImageData::Short(s) => s.to_float(),
        ImageData::Color(c) => c.to_float(),
    }
}

/// Convenience: convert a single `ImageData` to RGB.
pub fn to_rgb(proc: &ImageData) -> ColorProcessor {
    match proc {
        ImageData::Color(c) => c.clone(),
        ImageData::Byte(b) => b.to_rgb(),
        ImageData::Short(s) => s.to_rgb(),
        ImageData::Float(f) => f.to_rgb(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ij_core::ImagePlus;
    use crate::processor::{ByteProcessor, ColorProcessor, FloatProcessor, ShortProcessor};

    #[test]
    fn convert_rgb_to_gray8() {
        let cp = ColorProcessor::from_pixels(
            2,
            1,
            vec![0xffff_0000, 0xff00_ff00], // red, green
        );
        let mut imp = ImagePlus::from_image_data("test", ImageData::Color(cp));
        ImageConverter::new(&mut imp).convert_to_gray8();
        match imp.get_processor() {
            Some(ImageData::Byte(b)) => {
                // red(255,0,0)->76, green(0,255,0)->149 (approx)
                assert_eq!(b.pixels, vec![76, 149]);
            }
            _ => panic!("expected byte processor"),
        }
    }

    #[test]
    fn convert_gray16_to_gray8_scales() {
        let sp = ShortProcessor::from_pixels(2, 1, vec![0, 65535]);
        let mut imp = ImagePlus::from_image_data("test", ImageData::Short(sp));
        ImageConverter::new(&mut imp).convert_to_gray8();
        match imp.get_processor() {
            Some(ImageData::Byte(b)) => {
                assert_eq!(b.pixels, vec![0, 255]);
            }
            _ => panic!("expected byte processor"),
        }
    }

    #[test]
    fn convert_gray8_to_gray32() {
        let bp = ByteProcessor::from_pixels(2, 1, vec![10, 20]);
        let mut imp = ImagePlus::from_image_data("test", ImageData::Byte(bp));
        ImageConverter::new(&mut imp).convert_to_gray32();
        match imp.get_processor() {
            Some(ImageData::Float(f)) => {
                assert_eq!(f.pixels, vec![10.0, 20.0]);
            }
            _ => panic!("expected float processor"),
        }
    }

    #[test]
    fn convert_gray8_to_rgb() {
        let bp = ByteProcessor::from_pixels(2, 1, vec![0, 255]);
        let mut imp = ImagePlus::from_image_data("test", ImageData::Byte(bp));
        ImageConverter::new(&mut imp).convert_to_rgb();
        match imp.get_processor() {
            Some(ImageData::Color(c)) => {
                assert_eq!(c.pixels, vec![0xff000000, 0xffffffff]);
            }
            _ => panic!("expected color processor"),
        }
    }

    #[test]
    fn convert_gray32_to_gray16_scales() {
        let fp = FloatProcessor::from_pixels(2, 1, vec![0.0, 1.0]);
        let mut imp = ImagePlus::from_image_data("test", ImageData::Float(fp));
        ImageConverter::new(&mut imp).convert_to_gray16();
        match imp.get_processor() {
            Some(ImageData::Short(s)) => {
                assert_eq!(s.pixels, vec![0, 65535]);
            }
            _ => panic!("expected short processor"),
        }
    }

    #[test]
    fn idempotent_same_type() {
        let bp = ByteProcessor::from_pixels(2, 1, vec![5, 6]);
        let mut imp = ImagePlus::from_image_data("test", ImageData::Byte(bp));
        ImageConverter::new(&mut imp).convert_to_gray8();
        match imp.get_processor() {
            Some(ImageData::Byte(b)) => assert_eq!(b.pixels, vec![5, 6]),
            _ => panic!("still byte"),
        }
    }

    #[test]
    fn convenience_fns() {
        let sp = ShortProcessor::from_pixels(2, 1, vec![100, 200]);
        // default doScaling=true, range 0..65535 => both map near 0
        assert_eq!(to_gray8(&ImageData::Short(sp.clone())).pixels, vec![0, 0]);
        assert_eq!(
            to_gray32(&ImageData::Short(sp.clone())).pixels,
            vec![100.0, 200.0]
        );
        // with explicit display range, scaling maps to full 0..255
        let mut sp2 = ShortProcessor::from_pixels(2, 1, vec![100, 200]);
        sp2.set_min_and_max(100.0, 200.0);
        assert_eq!(to_gray8(&ImageData::Short(sp2)).pixels, vec![0, 255]);
    }
}