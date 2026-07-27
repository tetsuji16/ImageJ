//! Port of `ij.process.ColorSpaceConverter` — color space conversions.
//!
//! Implements RGB ⇄ XYZ ⇄ CIELAB ⇄ xyY and RGB ⇄ HSB, plus per-pixel
//! `rgb_to_lab` / `lab_to_rgb` helpers. Pure math, no AWT dependency.
//! (The Java original used `java.awt.Color` for HSB; we implement the
//! standard HSB algorithm directly to stay headless.)

/// Holds the conversion matrices and reference white point (D65 default).
#[derive(Debug, Clone)]
pub struct ColorSpaceConverter {
    /// Reference white point in XYZ (D65 by default).
    pub white_point: [f64; 3],
    /// sRGB → XYZ matrix.
    pub m: [[f64; 3]; 3],
    /// XYZ → sRGB matrix (inverse).
    pub mi: [[f64; 3]; 3],
}

impl Default for ColorSpaceConverter {
    fn default() -> Self {
        ColorSpaceConverter::new()
    }
}

impl ColorSpaceConverter {
    /// Creates a converter with the D65 white point (ImageJ default).
    pub fn new() -> Self {
        ColorSpaceConverter {
            white_point: [95.0429, 100.0, 108.8900],
            m: [
                [0.4124, 0.3576, 0.1805],
                [0.2126, 0.7152, 0.0722],
                [0.0193, 0.1192, 0.9505],
            ],
            mi: [
                [3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570],
            ],
        }
    }

    /// HSB → RGB. `h` is hue in [0,1] (= degrees/360), `s` saturation [0,1],
    /// `b` brightness/value [0,1]. Returns `[r, g, b]` each in 0..255.
    /// Mirrors `java.awt.Color.HSBtoRGB`.
    pub fn hsb_to_rgb(&self, h: f64, s: f64, b: f64) -> [u8; 3] {
        if s == 0.0 {
            let v = clamp255(b * 255.0);
            return [v, v, v];
        }
        let h = (h - h.floor()) * 6.0; // wrap to [0,6)
        let hi = h as i32;
        let f = h - hi as f64;
        let p = b * (1.0 - s);
        let q = b * (1.0 - s * f);
        let t = b * (1.0 - s * (1.0 - f));
        let (r, g, bl) = match hi {
            0 => (b, t, p),
            1 => (q, b, p),
            2 => (p, b, t),
            3 => (p, q, b),
            4 => (t, p, b),
            _ => (b, p, q),
        };
        [clamp255(r * 255.0), clamp255(g * 255.0), clamp255(bl * 255.0)]
    }

    /// RGB → HSB. Inputs `r,g,b` in 0..255. Returns `[h, s, b]` with
    /// `h,s,b` in [0,1]. Mirrors `java.awt.Color.RGBtoHSB`.
    pub fn rgb_to_hsb(&self, r: u8, g: u8, b: u8) -> [f64; 3] {
        let rn = r as f64 / 255.0;
        let gn = g as f64 / 255.0;
        let bn = b as f64 / 255.0;
        let max = rn.max(gn).max(bn);
        let min = rn.min(gn).min(bn);
        let delta = max - min;
        let brightness = max;
        let saturation = if max == 0.0 { 0.0 } else { delta / max };
        let hue = if delta == 0.0 {
            0.0
        } else if max == rn {
            (((gn - bn) / delta) % 6.0) / 6.0
        } else if max == gn {
            ((bn - rn) / delta + 2.0) / 6.0
        } else {
            ((rn - gn) / delta + 4.0) / 6.0
        };
        let hue = if hue < 0.0 { hue + 1.0 } else { hue };
        [hue, saturation, brightness]
    }

    /// RGB → XYZ (sRGB with gamma correction). Inputs in 0..255.
    pub fn rgb_to_xyz(&self, r: u8, g: u8, b: u8) -> [f64; 3] {
        let mut rn = r as f64 / 255.0;
        let mut gn = g as f64 / 255.0;
        let mut bn = b as f64 / 255.0;
        rn = srgb_to_linear(rn);
        gn = srgb_to_linear(gn);
        bn = srgb_to_linear(bn);
        rn *= 100.0;
        gn *= 100.0;
        bn *= 100.0;
        [
            self.m[0][0] * rn + self.m[0][1] * gn + self.m[0][2] * bn,
            self.m[1][0] * rn + self.m[1][1] * gn + self.m[1][2] * bn,
            self.m[2][0] * rn + self.m[2][1] * gn + self.m[2][2] * bn,
        ]
    }

    /// XYZ → RGB (sRGB with gamma correction). Returns `[r,g,b]` in 0..255.
    pub fn xyz_to_rgb(&self, x: f64, y: f64, z: f64) -> [u8; 3] {
        let xr = x / 100.0;
        let yr = y / 100.0;
        let zr = z / 100.0;
        let r = self.mi[0][0] * xr + self.mi[0][1] * yr + self.mi[0][2] * zr;
        let g = self.mi[1][0] * xr + self.mi[1][1] * yr + self.mi[1][2] * zr;
        let b = self.mi[2][0] * xr + self.mi[2][1] * yr + self.mi[2][2] * zr;
        [
            clamp255(linear_to_srgb(r) * 255.0),
            clamp255(linear_to_srgb(g) * 255.0),
            clamp255(linear_to_srgb(b) * 255.0),
        ]
    }

    /// XYZ → CIELAB. Uses the converter's white point.
    pub fn xyz_to_lab(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let xn = self.white_point[0];
        let yn = self.white_point[1];
        let zn = self.white_point[2];
        let xr = f_xyz(x / xn);
        let yr = f_xyz(y / yn);
        let zr = f_xyz(z / zn);
        [
            116.0 * yr - 16.0,
            500.0 * (xr - yr),
            200.0 * (yr - zr),
        ]
    }

    /// CIELAB → XYZ. Uses the converter's white point.
    pub fn lab_to_xyz(&self, l: f64, a: f64, b: f64) -> [f64; 3] {
        let yr = (l + 16.0) / 116.0;
        let xr = (a / 500.0) + yr;
        let zr = yr - (b / 200.0);
        let y = if yr.powi(3) > 0.008856 {
            yr.powi(3)
        } else {
            (yr - 16.0 / 116.0) / 7.787
        };
        let x = if xr.powi(3) > 0.008856 {
            xr.powi(3)
        } else {
            (xr - 16.0 / 116.0) / 7.787
        };
        let z = if zr.powi(3) > 0.008856 {
            zr.powi(3)
        } else {
            (zr - 16.0 / 116.0) / 7.787
        };
        [x * self.white_point[0], y * self.white_point[1], z * self.white_point[2]]
    }

    /// XYZ → xyY.
    pub fn xyz_to_xyy(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let sum = x + y + z;
        if sum == 0.0 {
            [0.3127, 0.3290, 100.0]
        } else {
            [x / sum, y / sum, y]
        }
    }

    /// xyY → XYZ.
    pub fn xyy_to_xyz(&self, x: f64, y: f64, yy: f64) -> [f64; 3] {
        if y == 0.0 {
            [0.0, 0.0, 0.0]
        } else {
            [(x * yy) / y, yy, ((1.0 - x - y) * yy) / y]
        }
    }

    /// Convenience: packed ARGB → `[L*, a*, b*]` (CIELAB, D65).
    pub fn rgb_to_lab(&self, rgb: u32) -> [f64; 3] {
        let r = ((rgb >> 16) & 0xff) as u8;
        let g = ((rgb >> 8) & 0xff) as u8;
        let b = (rgb & 0xff) as u8;
        let xyz = self.rgb_to_xyz(r, g, b);
        self.xyz_to_lab(xyz[0], xyz[1], xyz[2])
    }

    /// Convenience: `[L*, a*, b*]` → packed ARGB.
    pub fn lab_to_rgb(&self, l: f64, a: f64, b: f64) -> u32 {
        let xyz = self.lab_to_xyz(l, a, b);
        let rgb = self.xyz_to_rgb(xyz[0], xyz[1], xyz[2]);
        crate::processor::ColorProcessor::make_argb(255, rgb[0], rgb[1], rgb[2])
    }
}

/// sRGB companding: 0..1 linearized value.
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse sRGB companding: linear 0..1 → gamma-encoded 0..1.
fn linear_to_srgb(c: f64) -> f64 {
    let c = c.max(0.0);
    if c > 0.0031308 {
        (1.055 * c.powf(1.0 / 2.4)) - 0.055
    } else {
        c * 12.92
    }
}

/// Pivot function for XYZ → LAB (the 1/3 power / linear branch).
fn f_xyz(t: f64) -> f64 {
    if t > 0.008856 {
        t.powf(1.0 / 3.0)
    } else {
        7.787 * t + 16.0 / 116.0
    }
}

/// Clamp a float to a u8 in 0..255 (rounds).
fn clamp255(v: f64) -> u8 {
    if v < 0.0 {
        0
    } else if v > 255.0 {
        255
    } else {
        v.round() as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_xyz_white() {
        let c = ColorSpaceConverter::new();
        // pure white (255,255,255) → XYZ ≈ (95.04, 100, 108.89) for D65
        let xyz = c.rgb_to_xyz(255, 255, 255);
        assert!((xyz[0] - 95.04).abs() < 0.1);
        assert!((xyz[1] - 100.0).abs() < 0.1);
        assert!((xyz[2] - 108.89).abs() < 0.1);
    }

    #[test]
    fn xyz_roundtrip() {
        let c = ColorSpaceConverter::new();
        // (30,30,30) is inside the sRGB gamut, so the 8-bit roundtrip is tight
        let rgb = c.xyz_to_rgb(30.0, 30.0, 30.0);
        let xyz = c.rgb_to_xyz(rgb[0], rgb[1], rgb[2]);
        assert!((xyz[0] - 30.0).abs() < 1.0, "x={}", xyz[0]);
        assert!((xyz[1] - 30.0).abs() < 1.0, "y={}", xyz[1]);
        assert!((xyz[2] - 30.0).abs() < 1.0, "z={}", xyz[2]);
    }

    #[test]
    fn rgb_to_lab_black_is_zero() {
        let c = ColorSpaceConverter::new();
        let lab = c.rgb_to_lab(0xff00_0000); // black
        assert!(lab[0].abs() < 1e-6);
    }

    #[test]
    fn lab_roundtrip() {
        let c = ColorSpaceConverter::new();
        let rgb = 0xff80_40_20; // A=ff, R=0x80, G=0x40, B=0x20
        let lab = c.rgb_to_lab(rgb);
        let back = c.lab_to_rgb(lab[0], lab[1], lab[2]);
        assert_eq!(back, rgb);
    }

    #[test]
    fn hsb_red() {
        let c = ColorSpaceConverter::new();
        // hue 0, full sat/val → red
        let rgb = c.hsb_to_rgb(0.0, 1.0, 1.0);
        assert_eq!(rgb, [255, 0, 0]);
        let hsb = c.rgb_to_hsb(255, 0, 0);
        assert!((hsb[0]).abs() < 1e-6);
        assert!((hsb[1] - 1.0).abs() < 1e-6);
        assert!((hsb[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hsb_gray() {
        let c = ColorSpaceConverter::new();
        // zero saturation at value 0.4 → gray 102
        let rgb = c.hsb_to_rgb(0.3, 0.0, 0.4);
        assert_eq!(rgb, [102, 102, 102]);
    }
}
