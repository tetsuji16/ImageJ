//! Port of `ij.process.StackProcessor` — operations on an `ImageStack`.
//!
//! Geometric ops (flip/invert/rotate) apply to every slice in place; z-project
//! reduces a Z-range of slices to a single 2D `ImageData`. Mirrors the Java
//! methods (minus the AWT progress-bar / 3D-filter dependencies).

use crate::ij_core::ImageStack;
use crate::io::ImageData;
use crate::processor::{ByteProcessor, ColorProcessor, FloatProcessor, ShortProcessor};

/// Projection methods (subset of ImageJ's `ZProjector` methods).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    Average,
    Sum,
    Max,
    Min,
    /// Standard deviation across the projected slices (population SD).
    Sd,
}

/// Flips every slice horizontally (left-right).
pub fn flip_horizontal(stack: &mut ImageStack) {
    for i in 0..stack.size() {
        if let Some(slice) = stack.get_slice_mut(i) {
            match &mut slice.data {
                ImageData::Byte(bp) => *bp = bp.flip_horizontal(),
                ImageData::Short(sp) => *sp = sp.flip_horizontal(),
                ImageData::Float(fp) => *fp = fp.flip_horizontal(),
                ImageData::Color(cp) => *cp = cp.flip_horizontal(),
            }
        }
    }
}

/// Flips every slice vertically (top-bottom).
pub fn flip_vertical(stack: &mut ImageStack) {
    for i in 0..stack.size() {
        if let Some(slice) = stack.get_slice_mut(i) {
            match &mut slice.data {
                ImageData::Byte(bp) => *bp = bp.flip_vertical(),
                ImageData::Short(sp) => *sp = sp.flip_vertical(),
                ImageData::Float(fp) => *fp = fp.flip_vertical(),
                ImageData::Color(cp) => *cp = cp.flip_vertical(),
            }
        }
    }
}

/// Inverts every slice.
pub fn invert(stack: &mut ImageStack) {
    for i in 0..stack.size() {
        if let Some(slice) = stack.get_slice_mut(i) {
            if let ImageData::Byte(bp) = &mut slice.data {
                bp.invert();
            }
        }
    }
}

/// Applies a 256-entry lookup table to every 8-bit slice (no-op on others).
pub fn apply_table(stack: &mut ImageStack, table: &[u8; 256]) {
    for i in 0..stack.size() {
        if let Some(slice) = stack.get_slice_mut(i) {
            if let ImageData::Byte(bp) = &mut slice.data {
                bp.apply_table(table);
            }
        }
    }
}

/// Rotates every slice 90° clockwise. Swaps the stack width/height.
pub fn rotate90_cw(stack: &mut ImageStack) {
    for i in 0..stack.size() {
        if let Some(slice) = stack.get_slice_mut(i) {
            match &mut slice.data {
                ImageData::Byte(bp) => *bp = bp.rotate90_cw(),
                ImageData::Short(sp) => *sp = sp.rotate90_cw(),
                ImageData::Float(fp) => *fp = fp.rotate90_cw(),
                ImageData::Color(cp) => *cp = cp.rotate90_cw(),
            }
        }
    }
    std::mem::swap(&mut stack.width, &mut stack.height);
}

/// Z-projection over slices `start..=stop` (1-based slice numbers, inclusive).
/// Returns a single 2D `ImageData` of the stack's type with the same
/// width/height. Mirrors `ZProjector` (AVERAGE/SUM/MAX/MIN/SD). For 8/16-bit,
/// SUM is clamped to the type's range (a multi-slice sum can exceed it). For
/// RGB stacks all methods fall back to a per-channel average (ImageJ does not
/// define MAX/MIN/SUM/SD for RGB projection).
pub fn z_project(stack: &ImageStack, method: Projection, start: usize, stop: usize) -> ImageData {
    let lo = start.saturating_sub(1).min(stack.size().saturating_sub(1));
    let hi = stop.min(stack.size()).saturating_sub(1);
    let n = (hi - lo + 1).max(1);

    match &stack.slices[lo].data {
        ImageData::Byte(_) => {
            let w = stack.width;
            let h = stack.height;
            let mut out = ByteProcessor::new(w, h);
            for i in 0..w * h {
                let mut sum = 0u64;
                let mut sum_sq = 0.0f64;
                let mut mn = u8::MAX;
                let mut mx = 0u8;
                for s in lo..=hi {
                    if let ImageData::Byte(bp) = &stack.slices[s].data {
                        let v = bp.pixels[i] as f64;
                        sum += bp.pixels[i] as u64;
                        sum_sq += v * v;
                        if bp.pixels[i] < mn {
                            mn = bp.pixels[i];
                        }
                        if bp.pixels[i] > mx {
                            mx = bp.pixels[i];
                        }
                    }
                }
                let mean = sum as f64 / n as f64;
                out.pixels[i] = match method {
                    Projection::Average => mean.round().clamp(0.0, 255.0) as u8,
                    Projection::Sum => (sum as f64).clamp(0.0, 255.0) as u8,
                    Projection::Max => mx,
                    Projection::Min => mn,
                    Projection::Sd => {
                        let variance = (sum_sq / n as f64) - mean * mean;
                        let sd = if variance > 0.0 { variance.sqrt() } else { 0.0 };
                        sd.round().clamp(0.0, 255.0) as u8
                    }
                };
            }
            ImageData::Byte(out)
        }
        ImageData::Short(_) => {
            let w = stack.width;
            let h = stack.height;
            let mut out = ShortProcessor::new(w, h);
            for i in 0..w * h {
                let mut sum = 0u64;
                let mut sum_sq = 0.0f64;
                let mut mn = u16::MAX;
                let mut mx = 0u16;
                for s in lo..=hi {
                    if let ImageData::Short(sp) = &stack.slices[s].data {
                        let v = sp.pixels[i] as f64;
                        sum += sp.pixels[i] as u64;
                        sum_sq += v * v;
                        if sp.pixels[i] < mn {
                            mn = sp.pixels[i];
                        }
                        if sp.pixels[i] > mx {
                            mx = sp.pixels[i];
                        }
                    }
                }
                let mean = sum as f64 / n as f64;
                out.pixels[i] = match method {
                    Projection::Average => mean.round().clamp(0.0, 65535.0) as u16,
                    Projection::Sum => (sum as f64).clamp(0.0, 65535.0) as u16,
                    Projection::Max => mx,
                    Projection::Min => mn,
                    Projection::Sd => {
                        let variance = (sum_sq / n as f64) - mean * mean;
                        let sd = if variance > 0.0 { variance.sqrt() } else { 0.0 };
                        sd.round().clamp(0.0, 65535.0) as u16
                    }
                };
            }
            ImageData::Short(out)
        }
        ImageData::Float(_) => {
            let w = stack.width;
            let h = stack.height;
            let mut out = FloatProcessor::new(w, h);
            for i in 0..w * h {
                let mut sum = 0.0f64;
                let mut sum_sq = 0.0f64;
                let mut mn = f64::INFINITY;
                let mut mx = f64::NEG_INFINITY;
                for s in lo..=hi {
                    if let ImageData::Float(fp) = &stack.slices[s].data {
                        let v = fp.pixels[i] as f64;
                        sum += v;
                        sum_sq += v * v;
                        if v < mn {
                            mn = v;
                        }
                        if v > mx {
                            mx = v;
                        }
                    }
                }
                let mean = sum / n as f64;
                out.pixels[i] = match method {
                    Projection::Average => mean as f32,
                    Projection::Sum => sum as f32,
                    Projection::Max => mx as f32,
                    Projection::Min => mn as f32,
                    Projection::Sd => {
                        let variance = (sum_sq / n as f64) - mean * mean;
                        let sd = if variance > 0.0 { variance.sqrt() } else { 0.0 };
                        sd as f32
                    }
                };
            }
            ImageData::Float(out)
        }
        ImageData::Color(_) => {
            let w = stack.width;
            let h = stack.height;
            let mut out = ColorProcessor::new(w, h);
            for i in 0..w * h {
                let mut sr = 0u64;
                let mut sg = 0u64;
                let mut sb = 0u64;
                for s in lo..=hi {
                    if let ImageData::Color(cp) = &stack.slices[s].data {
                        let p = cp.pixels[i];
                        sr += ((p >> 16) & 0xff) as u64;
                        sg += ((p >> 8) & 0xff) as u64;
                        sb += (p & 0xff) as u64;
                    }
                }
                let r = (sr / n as u64) as u8;
                let g = (sg / n as u64) as u8;
                let b = (sb / n as u64) as u8;
                out.pixels[i] = ColorProcessor::make_argb(255, r, g, b);
            }
            ImageData::Color(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::ByteProcessor;

    fn two_slice_stack() -> ImageStack {
        let mut stack = ImageStack::new(2, 2);
        stack.add_slice(
            None,
            ImageData::Byte(ByteProcessor::from_pixels(2, 2, vec![1, 2, 3, 4])),
        );
        stack.add_slice(
            None,
            ImageData::Byte(ByteProcessor::from_pixels(2, 2, vec![5, 6, 7, 8])),
        );
        stack
    }

    #[test]
    fn flip_h_stack() {
        let mut stack = two_slice_stack();
        flip_horizontal(&mut stack);
        let a = stack.get_slice(0).unwrap();
        if let ImageData::Byte(bp) = &a.data {
            assert_eq!(bp.pixels, vec![2, 1, 4, 3]);
        } else {
            panic!();
        }
    }

    #[test]
    fn z_project_max() {
        let stack = two_slice_stack();
        let proj = z_project(&stack, Projection::Max, 1, 2);
        if let ImageData::Byte(bp) = &proj {
            assert_eq!(bp.pixels, vec![5, 6, 7, 8]);
        } else {
            panic!();
        }
    }

    #[test]
    fn z_project_average() {
        let stack = two_slice_stack();
        let proj = z_project(&stack, Projection::Average, 1, 2);
        if let ImageData::Byte(bp) = &proj {
            assert_eq!(bp.pixels, vec![3, 4, 5, 6]); // (1+5)/2=3, (2+6)/2=4...
        } else {
            panic!();
        }
    }

    #[test]
    fn z_project_min() {
        let stack = two_slice_stack();
        let proj = z_project(&stack, Projection::Min, 1, 2);
        if let ImageData::Byte(bp) = &proj {
            assert_eq!(bp.pixels, vec![1, 2, 3, 4]);
        } else {
            panic!();
        }
    }

    #[test]
    fn z_project_sum() {
        // regression: Sum must sum, not average
        let stack = two_slice_stack();
        let proj = z_project(&stack, Projection::Sum, 1, 2);
        if let ImageData::Byte(bp) = &proj {
            assert_eq!(bp.pixels, vec![6, 8, 10, 12]); // 1+5, 2+6, 3+7, 4+8
        } else {
            panic!();
        }
    }

    #[test]
    fn z_project_sd() {
        // regression: Sd must be the actual standard deviation, not the average
        let stack = two_slice_stack();
        let proj = z_project(&stack, Projection::Sd, 1, 2);
        if let ImageData::Byte(bp) = &proj {
            // values 1,5 -> mean 3, var 4, sd 2 (and similarly 2,6 / 3,7 / 4,8)
            assert_eq!(bp.pixels, vec![2, 2, 2, 2]);
        } else {
            panic!();
        }
    }
}
