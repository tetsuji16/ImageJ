//! Port of `ij.process.LUT` — an indexed color lookup table (256 RGB entries).
//!
//! Mirrors the Java `LUT` (an `IndexColorModel` subclass) minus AWT deps.
//! LUTs map a pixel value (0-255) to an RGB color, used for display and
//! pseudo-coloring grayscale images.

/// Lookup table: 256 RGB entries + display min/max bounds.
#[derive(Debug, Clone)]
pub struct Lut {
    /// Red channel for each of the 256 indices (stored as u8).
    pub reds: [u8; 256],
    /// Green channel for each of the 256 indices.
    pub greens: [u8; 256],
    /// Blue channel for each of the 256 indices.
    pub blues: [u8; 256],
    /// Displayed lower bound (pixel value mapped to index 0).
    pub min: f64,
    /// Displayed upper bound (pixel value mapped to index 255).
    pub max: f64,
}

/// Standard lookup table names (mirrors common ImageJ presets).
pub mod preset {
    /// Grayscale ramp (identity: index i -> (i,i,i)).
    pub fn grayscale() -> super::Lut {
        super::Lut::grayscale()
    }
    /// Fire / thermal lookup table.
    pub fn fire() -> super::Lut {
        super::Lut::fire()
    }
    /// ICE lookup table (blue->white->red).
    pub fn ice() -> super::Lut {
        super::Lut::ice()
    }
    /// Spectrum / rainbow lookup table.
    pub fn spectrum() -> super::Lut {
        super::Lut::spectrum()
    }
}

impl Lut {
    /// Creates a LUT from red, green, blue byte slices (each length 256).
    pub fn new(reds: [u8; 256], greens: [u8; 256], blues: [u8; 256]) -> Self {
        Lut {
            reds,
            greens,
            blues,
            min: 0.0,
            max: 255.0,
        }
    }

    /// Grayscale ramp: index i -> (i, i, i).
    pub fn grayscale() -> Self {
        let mut reds = [0u8; 256];
        let mut greens = [0u8; 256];
        let mut blues = [0u8; 256];
        for i in 0..256u32 {
            let v = i as u8;
            reds[i as usize] = v;
            greens[i as usize] = v;
            blues[i as usize] = v;
        }
        Lut {
            reds,
            greens,
            blues,
            min: 0.0,
            max: 255.0,
        }
    }

    /// Fire lookup table (black -> red -> yellow -> white).
    pub fn fire() -> Self {
        let mut reds = [0u8; 256];
        let mut greens = [0u8; 256];
        let mut blues = [0u8; 256];
        for i in 0..256 {
            let x = i as f64 / 255.0;
            reds[i] = (255.0 * x.min(1.0)) as u8;
            greens[i] = (255.0 * (2.0 * x - 1.0).max(0.0).min(1.0)) as u8;
            blues[i] = (255.0 * (3.0 * x - 2.0).max(0.0).min(1.0)) as u8;
        }
        Lut {
            reds,
            greens,
            blues,
            min: 0.0,
            max: 255.0,
        }
    }

    /// ICE lookup table (cyan/blue -> white -> red/yellow).
    pub fn ice() -> Self {
        let mut reds = [0u8; 256];
        let mut greens = [0u8; 256];
        let mut blues = [0u8; 256];
        for i in 0..256 {
            let x = i as f64 / 255.0;
            reds[i] = (255.0 * (1.5 * x - 0.5).max(0.0).min(1.0)) as u8;
            greens[i] = (255.0 * (1.5 * x - 0.25).max(0.0).min(1.0)) as u8;
            blues[i] = (255.0 * (2.0 * x).min(1.0)) as u8;
        }
        Lut {
            reds,
            greens,
            blues,
            min: 0.0,
            max: 255.0,
        }
    }

    /// Spectrum / rainbow lookup table.
    pub fn spectrum() -> Self {
        let mut reds = [0u8; 256];
        let mut greens = [0u8; 256];
        let mut blues = [0u8; 256];
        for i in 0..256 {
            let h = i as f64 / 255.0 * 360.0; // hue 0..360
            let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
            reds[i] = r;
            greens[i] = g;
            blues[i] = b;
        }
        Lut {
            reds,
            greens,
            blues,
            min: 0.0,
            max: 255.0,
        }
    }

    /// Returns the packed RGB value (0xRRGGBB) for the given index.
    pub fn get_rgb(&self, index: usize) -> u32 {
        let i = index.min(255);
        ((self.reds[i] as u32) << 16) | ((self.greens[i] as u32) << 8) | (self.blues[i] as u32)
    }

    /// Looks up the RGB color for a pixel value (clamped to [min,max] range,
    /// then scaled to 0-255 index). Returns a packed 32-bit ARGB value
    /// (0xFF in the high byte).
    pub fn apply(&self, value: f64) -> u32 {
        if self.max <= self.min {
            return 0xff000000 | self.get_rgb(value.round().clamp(0.0, 255.0) as usize);
        }
        let t = (value - self.min) / (self.max - self.min);
        let idx = (t * 255.0).round().clamp(0.0, 255.0) as usize;
        0xff000000 | self.get_rgb(idx)
    }

    /// Creates an inverted copy of this LUT (reversed ramp).
    pub fn create_inverted_lut(&self) -> Lut {
        let mut reds = [0u8; 256];
        let mut greens = [0u8; 256];
        let mut blues = [0u8; 256];
        for i in 0..256 {
            let j = 255 - i;
            reds[i] = self.reds[j];
            greens[i] = self.greens[j];
            blues[i] = self.blues[j];
        }
        Lut {
            reds,
            greens,
            blues,
            min: self.min,
            max: self.max,
        }
    }

    /// Builds a LUT from a single base color: index i maps to
    /// (i*R/255, i*G/255, i*B/255). Mirrors `createLutFromColor`.
    pub fn from_color(r: u8, g: u8, b: u8) -> Lut {
        let mut reds = [0u8; 256];
        let mut greens = [0u8; 256];
        let mut blues = [0u8; 256];
        let r_incr = r as f64 / 255.0;
        let g_incr = g as f64 / 255.0;
        let b_incr = b as f64 / 255.0;
        for i in 0..256 {
            let x = i as f64;
            reds[i] = (x * r_incr) as u8;
            greens[i] = (x * g_incr) as u8;
            blues[i] = (x * b_incr) as u8;
        }
        Lut {
            reds,
            greens,
            blues,
            min: 0.0,
            max: 255.0,
        }
    }

    /// Maps this LUT onto an 8-bit grayscale image, returning a 32-bit ARGB buffer.
    pub fn map_image(&self, gray: &[u8]) -> Vec<u32> {
        gray.iter().map(|&v| 0xff000000 | self.get_rgb(v as usize)).collect()
    }

    /// Returns the three channel arrays concatenated as [r0..r255, g0..g255, b0..b255].
    pub fn get_bytes(&self) -> [u8; 768] {
        let mut bytes = [0u8; 768];
        for i in 0..256 {
            bytes[i] = self.reds[i];
            bytes[256 + i] = self.greens[i];
            bytes[512 + i] = self.blues[i];
        }
        bytes
    }

    /// Sets the display bounds.
    pub fn set_min_and_max(&mut self, min: f64, max: f64) {
        self.min = min;
        self.max = max;
    }
}

/// HSV → RGB conversion (h in degrees 0..360, s/v in 0..1).
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    (
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grayscale_identity() {
        let lut = Lut::grayscale();
        assert_eq!(lut.get_rgb(0), 0x000000);
        assert_eq!(lut.get_rgb(128), 0x808080);
        assert_eq!(lut.get_rgb(255), 0xffffff);
        assert_eq!(lut.apply(0.0), 0xff000000);
        assert_eq!(lut.apply(255.0), 0xffffffff);
    }

    #[test]
    fn fire_ramp_increases_red() {
        let lut = Lut::fire();
        // index 255 should be white
        assert_eq!(lut.get_rgb(255), 0xffffff);
        // index 0 should be black
        assert_eq!(lut.get_rgb(0), 0x000000);
        // red monotonically increases
        assert!(lut.reds[100] > lut.reds[50]);
    }

    #[test]
    fn inverted_lut_reverses() {
        let lut = Lut::grayscale();
        let inv = lut.create_inverted_lut();
        assert_eq!(inv.get_rgb(0), 0xffffff);
        assert_eq!(inv.get_rgb(255), 0x000000);
    }

    #[test]
    fn from_color_scales() {
        let lut = Lut::from_color(255, 0, 0); // pure red
        assert_eq!(lut.get_rgb(0), 0x000000);
        assert_eq!(lut.get_rgb(255), 0xff0000);
        assert_eq!(lut.get_rgb(128), 0x800000);
    }

    #[test]
    fn apply_scales_with_min_max() {
        let mut lut = Lut::grayscale();
        lut.set_min_and_max(0.0, 127.0);
        // value 127 -> top of range -> white
        assert_eq!(lut.apply(127.0), 0xffffffff);
        // value 63.5 -> middle -> gray
        assert_eq!(lut.apply(63.5), 0xff808080);
    }

    #[test]
    fn map_image_produces_argb() {
        let lut = Lut::grayscale();
        let out = lut.map_image(&[0, 128, 255]);
        assert_eq!(out, vec![0xff000000, 0xff808080, 0xffffffff]);
    }

    #[test]
    fn get_bytes_layout() {
        let lut = Lut::grayscale();
        let bytes = lut.get_bytes();
        assert_eq!(bytes.len(), 768);
        assert_eq!(bytes[0], 0); // r[0]
        assert_eq!(bytes[255], 255); // r[255]
        assert_eq!(bytes[256], 0); // g[0]
        assert_eq!(bytes[511], 255); // g[255]
        assert_eq!(bytes[512], 0); // b[0]
        assert_eq!(bytes[767], 255); // b[255]
    }

    #[test]
    fn spectrum_full_cycle() {
        let lut = Lut::spectrum();
        // hue 0 and 360 both map to red (approximately)
        assert!(lut.reds[0] > 200);
        assert!(lut.reds[255] > 200);
        // middle hue (180 = cyan) should have low red
        assert!(lut.reds[128] < 100);
    }
}