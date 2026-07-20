//! Port of `ij.process.Blitter` (interface + blend-mode constants).
//!
//! Original (Java):
//! ```java
//! public interface Blitter {
//!     int COPY = 0;
//!     int COPY_INVERTED = 1;
//!     int COPY_TRANSPARENT = 2;
//!     int ADD = 3;
//!     int SUBTRACT = 4;
//!     int MULTIPLY = 5;
//!     int DIVIDE = 6;
//!     int AVERAGE = 7;
//!     int DIFFERENCE = 8;
//!     int AND = 9;
//!     int OR = 10;
//!     int XOR = 11;
//!     int MIN = 12;
//!     int MAX = 13;
//!     int COPY_ZERO_TRANSPARENT = 14;
//! }
//! ```
//!
//! In Rust we model the modes as an enum (type-safe, no magic integers).
//! The per-channel arithmetic is ported from `ij.process.ByteBlitter`,
//! which implements the actual `copyBits` math for 8-bit images.

/// Blend modes for combining a source pixel into a destination pixel.
///
/// Mirrors the `Blitter.*` integer constants in ImageJ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlendMode {
    /// `dst = src`
    Copy = 0,
    /// `dst = 255 - src` (8-bit and RGB)
    CopyInverted = 1,
    /// Copies with white pixels transparent.
    CopyTransparent = 2,
    /// `dst = dst + src`
    Add = 3,
    /// `dst = dst - src`
    Subtract = 4,
    /// `dst = src * src`
    Multiply = 5,
    /// `dst = dst / src`
    Divide = 6,
    /// `dst = (dst + src) / 2`
    Average = 7,
    /// `dst = |dst - src|`
    Difference = 8,
    /// `dst = dst & src`
    And = 9,
    /// `dst = dst | src`
    Or = 10,
    /// `dst = dst ^ src`
    Xor = 11,
    /// `dst = min(dst, src)`
    Min = 12,
    /// `dst = max(dst, src)`
    Max = 13,
    /// Copies with zero pixels transparent.
    CopyZeroTransparent = 14,
}

impl BlendMode {
    /// Returns the mode for a raw ImageJ integer code, if valid.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Copy),
            1 => Some(Self::CopyInverted),
            2 => Some(Self::CopyTransparent),
            3 => Some(Self::Add),
            4 => Some(Self::Subtract),
            5 => Some(Self::Multiply),
            6 => Some(Self::Divide),
            7 => Some(Self::Average),
            8 => Some(Self::Difference),
            9 => Some(Self::And),
            10 => Some(Self::Or),
            11 => Some(Self::Xor),
            12 => Some(Self::Min),
            13 => Some(Self::Max),
            14 => Some(Self::CopyZeroTransparent),
            _ => None,
        }
    }
}

/// Blends a single 8-bit source channel into a destination channel.
///
/// Ported 1:1 from `ij.process.ByteBlitter.copyBits` (the `switch (mode)`
/// body, applied per pixel). Java uses signed `byte` masked with `&255`;
/// here `dst`/`src` are already unsigned `u8`, so the arithmetic is direct.
///
/// Clamping matches ImageJ exactly:
/// - `Add` / `Multiply`: overflow saturates to 255.
/// - `Subtract`: underflow saturates to 0.
/// - `Difference`: absolute value (never negative).
/// - `Divide`: division by zero yields 255.
pub fn blend_channel(dst: u8, src: u8, mode: BlendMode) -> u8 {
    match mode {
        BlendMode::Copy => src,
        BlendMode::CopyInverted => 255 - src,
        BlendMode::CopyTransparent => src, // transparent handling is done by the caller
        BlendMode::CopyZeroTransparent => src, // transparent handling is done by the caller
        BlendMode::Add => {
            let s = dst as u16 + src as u16;
            if s > 255 {
                255
            } else {
                s as u8
            }
        }
        BlendMode::Subtract => {
            let s = dst as i16 - src as i16;
            if s < 0 {
                0
            } else {
                s as u8
            }
        }
        BlendMode::Multiply => {
            let s = dst as u16 * src as u16;
            if s > 255 {
                255
            } else {
                s as u8
            }
        }
        BlendMode::Divide => {
            if src == 0 {
                255
            } else {
                dst / src
            }
        }
        BlendMode::Average => ((dst as u16 + src as u16) / 2) as u8,
        BlendMode::Difference => {
            let s = dst as i16 - src as i16;
            s.unsigned_abs() as u8
        }
        BlendMode::And => dst & src,
        BlendMode::Or => dst | src,
        BlendMode::Xor => dst ^ src,
        BlendMode::Min => dst.min(src),
        BlendMode::Max => dst.max(src),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_code_roundtrip() {
        for code in 0u8..=14 {
            let m = BlendMode::from_code(code);
            assert!(m.is_some(), "code {code} should be valid");
            assert_eq!(m.unwrap() as u8, code);
        }
        assert_eq!(BlendMode::from_code(15), None);
        assert_eq!(BlendMode::from_code(255), None);
    }

    #[test]
    fn copy_and_invert() {
        assert_eq!(blend_channel(10, 200, BlendMode::Copy), 200);
        assert_eq!(blend_channel(10, 200, BlendMode::CopyInverted), 55);
    }

    #[test]
    fn add_saturates() {
        assert_eq!(blend_channel(200, 100, BlendMode::Add), 255);
        assert_eq!(blend_channel(100, 50, BlendMode::Add), 150);
    }

    #[test]
    fn subtract_underflow_zero() {
        assert_eq!(blend_channel(30, 80, BlendMode::Subtract), 0);
        assert_eq!(blend_channel(80, 30, BlendMode::Subtract), 50);
    }

    #[test]
    fn multiply_saturates() {
        assert_eq!(blend_channel(200, 200, BlendMode::Multiply), 255);
        assert_eq!(blend_channel(10, 10, BlendMode::Multiply), 100);
    }

    #[test]
    fn divide_by_zero_is_255() {
        assert_eq!(blend_channel(100, 0, BlendMode::Divide), 255);
        assert_eq!(blend_channel(100, 10, BlendMode::Divide), 10);
    }

    #[test]
    fn difference_is_abs() {
        assert_eq!(blend_channel(30, 80, BlendMode::Difference), 50);
        assert_eq!(blend_channel(80, 30, BlendMode::Difference), 50);
    }

    #[test]
    fn average_floors() {
        // (100+101)/2 = 100 in integer arithmetic, matching Java's integer divide
        assert_eq!(blend_channel(100, 101, BlendMode::Average), 100);
        assert_eq!(blend_channel(0, 255, BlendMode::Average), 127);
    }

    #[test]
    fn bitwise_modes() {
        assert_eq!(blend_channel(0b1100, 0b1010, BlendMode::And), 0b1000);
        assert_eq!(blend_channel(0b1100, 0b1010, BlendMode::Or), 0b1110);
        assert_eq!(blend_channel(0b1100, 0b1010, BlendMode::Xor), 0b0110);
    }

    #[test]
    fn min_max() {
        assert_eq!(blend_channel(40, 90, BlendMode::Min), 40);
        assert_eq!(blend_channel(40, 90, BlendMode::Max), 90);
    }
}
