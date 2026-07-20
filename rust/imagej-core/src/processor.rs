//! Port of `ij.process.ByteProcessor` — the 8-bit image processor.
//!
//! This is the first concrete `ImageProcessor` implementation in the Rust
//! port. The Java `ImageProcessor` is a large abstract class; we start with
//! the self-contained, pure-data parts of `ByteProcessor` (pixel buffer,
//! width/height, ROI, get/set, duplicate, histogram, min/max) so they can be
//! unit-tested 1:1 against the Java reference. Drawing, LUT/color-model,
//! and AWT-dependent methods (`createImage`, `getBufferedImage`) are deferred.
//!
//! Java notes mirrored here:
//! - `pixels` is `byte[]`; Java reads it masked with `&0xff` (unsigned).
//!   We store `Vec<u8>`, so no masking is needed.
//! - `min`/`max` are the *displayed* LUT range (init 0/255), set via
//!   `setMinAndMax`. They are NOT the pixel data min/max.
//! - `getPixel` returns 0 outside the bounds; `setPixel` ignores out-of-bounds.

/// An 8-bit grayscale image processor.
///
/// Mirrors `ij.process.ByteProcessor`. Pixel data is a flat `width*height`
/// row-major `Vec<u8>`.
#[derive(Debug, Clone)]
pub struct ByteProcessor {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,

    /// Displayed LUT min (Java `min`), init 0.
    pub min: i32,
    /// Displayed LUT max (Java `max`), init 255.
    pub max: i32,

    // ROI (region of interest). Defaults to the full image.
    roi_x: usize,
    roi_y: usize,
    roi_width: usize,
    roi_height: usize,
}

impl ByteProcessor {
    /// Creates a blank `width x height` image, zero-initialized.
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        ByteProcessor {
            width,
            height,
            pixels: vec![0u8; n],
            min: 0,
            max: 255,
            roi_x: 0,
            roi_y: 0,
            roi_width: width,
            roi_height: height,
        }
    }

    /// Creates a processor that takes ownership of `pixels` (length must be
    /// `width*height`).
    pub fn from_pixels(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        assert_eq!(pixels.len(), width * height, "pixel length mismatch");
        let mut bp = Self::new(width, height);
        bp.pixels = pixels;
        bp
    }

    /// Returns the pixel value (0-255) at the given index, no bounds check.
    /// Mirrors `get(int index)`.
    #[inline]
    pub fn get_index(&self, index: usize) -> u8 {
        self.pixels[index]
    }

    /// Sets the pixel at `index` (no bounds check). Mirrors `set(int, int)`.
    #[inline]
    pub fn set_index(&mut self, index: usize, value: u8) {
        self.pixels[index] = value;
    }

    /// Returns the pixel value at (x,y), or 0 if out of bounds.
    /// Mirrors `getPixel(int, int)`.
    pub fn get_pixel(&self, x: i32, y: i32) -> u8 {
        if x >= 0 && (x as usize) < self.width && y >= 0 && (y as usize) < self.height {
            self.pixels[y as usize * self.width + x as usize]
        } else {
            0
        }
    }

    /// Sets the pixel at (x,y); ignored if out of bounds. Values are clamped
    /// to 0-255. Mirrors `putPixel(int, int, int)` (clamp) / `set(int,int,int)`.
    pub fn set_pixel(&mut self, x: i32, y: i32, value: i32) {
        if x >= 0 && (x as usize) < self.width && y >= 0 && (y as usize) < self.height {
            let v = if value > 255 {
                255
            } else if value < 0 {
                0
            } else {
                value
            };
            self.pixels[y as usize * self.width + x as usize] = v as u8;
        }
    }

    /// Returns the calibrated pixel value at (x,y), or `f64::NAN` if out of
    /// bounds. Our port has no calibration table yet, so it equals the raw
    /// value. Mirrors `getPixelValue(int, int)`.
    pub fn get_pixel_value(&self, x: i32, y: i32) -> f64 {
        if x >= 0 && (x as usize) < self.width && y >= 0 && (y as usize) < self.height {
            self.pixels[y as usize * self.width + x as usize] as f64
        } else {
            f64::NAN
        }
    }

    /// Stores `value` at (x,y), clamped to 0-255; ignored out of bounds.
    /// Mirrors `putPixelValue(int, int, double)`.
    pub fn put_pixel_value(&mut self, x: i32, y: i32, value: f64) {
        if x >= 0 && (x as usize) < self.width && y >= 0 && (y as usize) < self.height {
            let v = if value > 255.0 {
                255.0
            } else if value < 0.0 {
                0.0
            } else {
                value
            };
            self.pixels[y as usize * self.width + x as usize] = (v + 0.5) as u8;
        }
    }

    /// Returns a deep copy of this processor (same dimensions, min/max, ROI,
    /// and pixel data). Mirrors `duplicate()`.
    pub fn duplicate(&self) -> Self {
        ByteProcessor {
            width: self.width,
            height: self.height,
            pixels: self.pixels.clone(),
            min: self.min,
            max: self.max,
            roi_x: self.roi_x,
            roi_y: self.roi_y,
            roi_width: self.roi_width,
            roi_height: self.roi_height,
        }
    }

    /// Returns a new blank `ByteProcessor` of the given size. Mirrors
    /// `createProcessor(int, int)` (without color model / interpolation state,
    /// which are deferred).
    pub fn create_processor(&self, width: usize, height: usize) -> Self {
        let mut bp = ByteProcessor::new(width, height);
        bp.min = self.min;
        bp.max = self.max;
        bp
    }

    /// 256-bin histogram over the current ROI. Mirrors `getHistogram()`
    /// (no mask). Delegates to the pure `histogram_8bit` helper over the ROI
    /// sub-region, keeping parity with `ByteStatistics`.
    pub fn get_histogram(&self) -> [u32; 256] {
        let mut hist = [0u32; 256];
        for y in self.roi_y..(self.roi_y + self.roi_height) {
            let base = y * self.width + self.roi_x;
            for x in 0..self.roi_width {
                hist[self.pixels[base + x] as usize] += 1;
            }
        }
        hist
    }

    /// Smallest displayed value (LUT min). Mirrors `getMin()`.
    pub fn get_min(&self) -> i32 {
        self.min
    }

    /// Largest displayed value (LUT max). Mirrors `getMax()`.
    pub fn get_max(&self) -> i32 {
        self.max
    }

    /// Sets the displayed LUT range [min, max]. Mirrors `setMinAndMax`:
    /// if `max < min`, the call is a no-op (as in Java).
    pub fn set_min_and_max(&mut self, min: f64, max: f64) {
        if max < min {
            return;
        }
        self.min = min.round() as i32;
        self.max = max.round() as i32;
    }

    /// Sets the ROI. Mirrors `setRoi(int, int, int, int)`.
    pub fn set_roi(&mut self, x: usize, y: usize, w: usize, h: usize) {
        self.roi_x = x;
        self.roi_y = y;
        self.roi_width = w;
        self.roi_height = h;
    }

    /// Resets the ROI to the full image. Mirrors `resetRoi()`.
    pub fn reset_roi(&mut self) {
        self.roi_x = 0;
        self.roi_y = 0;
        self.roi_width = self.width;
        self.roi_height = self.height;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_blank() {
        let bp = ByteProcessor::new(4, 3);
        assert_eq!(bp.width, 4);
        assert_eq!(bp.height, 3);
        assert_eq!(bp.pixels.len(), 12);
        assert!(bp.pixels.iter().all(|&v| v == 0));
        assert_eq!(bp.get_min(), 0);
        assert_eq!(bp.get_max(), 255);
    }

    #[test]
    fn get_set_pixel_bounds() {
        let mut bp = ByteProcessor::new(3, 3);
        bp.set_pixel(1, 1, 200);
        assert_eq!(bp.get_pixel(1, 1), 200);
        // out of bounds: get -> 0, set -> ignored
        assert_eq!(bp.get_pixel(-1, 0), 0);
        assert_eq!(bp.get_pixel(3, 0), 0);
        bp.set_pixel(99, 99, 123);
        assert_eq!(bp.get_pixel(0, 0), 0);
    }

    #[test]
    fn set_pixel_clamps() {
        let mut bp = ByteProcessor::new(2, 2);
        bp.set_pixel(0, 0, 300);
        bp.set_pixel(1, 0, -50);
        assert_eq!(bp.get_pixel(0, 0), 255);
        assert_eq!(bp.get_pixel(1, 0), 0);
    }

    #[test]
    fn put_pixel_value_rounds_and_clamps() {
        let mut bp = ByteProcessor::new(2, 2);
        // (10.4 + 0.5) = 10.9 -> 10
        bp.put_pixel_value(0, 0, 10.4);
        assert_eq!(bp.get_pixel(0, 0), 10);
        // (10.6 + 0.5) = 11.1 -> 11
        bp.put_pixel_value(1, 0, 10.6);
        assert_eq!(bp.get_pixel(1, 0), 11);
        bp.put_pixel_value(0, 1, 999.0);
        assert_eq!(bp.get_pixel(0, 1), 255);
    }

    #[test]
    fn duplicate_is_independent() {
        let mut bp = ByteProcessor::new(2, 2);
        bp.set_pixel(0, 0, 7);
        let mut d = bp.duplicate();
        d.set_pixel(0, 0, 99);
        assert_eq!(bp.get_pixel(0, 0), 7); // original unchanged
        assert_eq!(d.get_pixel(0, 0), 99);
    }

    #[test]
    fn histogram_over_roi() {
        let mut bp = ByteProcessor::from_pixels(
            4,
            2,
            vec![
                0, 0, 5, 5, // row 0
                5, 5, 9, 9, // row 1
            ],
        );
        let full = bp.get_histogram();
        assert_eq!(full[0], 2);
        assert_eq!(full[5], 4);
        assert_eq!(full[9], 2);
        // ROI = top row only -> only 0,0,5,5
        bp.set_roi(0, 0, 4, 1);
        let roi = bp.get_histogram();
        assert_eq!(roi[0], 2);
        assert_eq!(roi[5], 2);
        assert_eq!(roi[9], 0);
    }

    #[test]
    fn set_min_max_noop_when_reversed() {
        let mut bp = ByteProcessor::new(2, 2);
        bp.set_min_and_max(100.0, 50.0); // max < min -> ignored
        assert_eq!(bp.get_min(), 0);
        assert_eq!(bp.get_max(), 255);
        bp.set_min_and_max(10.0, 20.0);
        assert_eq!(bp.get_min(), 10);
        assert_eq!(bp.get_max(), 20);
    }

    #[test]
    fn create_processor_copies_min_max() {
        let mut bp = ByteProcessor::new(2, 2);
        bp.set_min_and_max(40.0, 200.0);
        let p2 = bp.create_processor(5, 5);
        assert_eq!(p2.get_min(), 40);
        assert_eq!(p2.get_max(), 200);
        assert_eq!(p2.width, 5);
    }
}
