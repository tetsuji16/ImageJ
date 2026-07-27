//! Port of `ij.process.BinaryProcessor` — binary image processing.
//!
//! Implements `outline()` and `skeletonize()` (thinning) using the same
//! 256-entry lookup tables as the Java reference. Operates in-place on a
//! `ByteProcessor` whose pixels are 0 (background) or 255 (foreground).

use crate::processor::ByteProcessor;

/// First skeletonization lookup table (indexed by 3x3 neighborhood bitmask).
static TABLE: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
];

/// Second skeletonization lookup table (removes "stuck" pixels).
static TABLE2: [u8; 256] = [
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Binary image processor. Wraps a `ByteProcessor` containing 0/255 pixels.
pub struct BinaryProcessor<'a> {
    pub parent: &'a mut ByteProcessor,
    foreground: u8,
}

impl<'a> BinaryProcessor<'a> {
    /// Creates a binary processor from a byte processor (pixels must be 0 or 255).
    pub fn new(parent: &'a mut ByteProcessor) -> Self {
        BinaryProcessor {
            parent,
            foreground: 255,
        }
    }

    /// Sets the foreground value (255 or 0).
    pub fn set_foreground(&mut self, fg: u8) {
        assert!(fg == 255 || fg == 0, "foreground must be 255 or 0");
        self.foreground = fg;
    }

    /// Converts binary objects to single-pixel outlines (in-place).
    pub fn outline(&mut self) {
        let (w, h) = (self.parent.width, self.parent.height);
        let snapshot = self.parent.pixels.clone();
        let bg = 255 - self.foreground;
        for y in 0..h {
            for x in 0..w {
                let v = snapshot[y * w + x];
                if v != bg {
                    let p1 = get_u8(&snapshot, w as i32, h as i32, x as i32 - 1, y as i32 - 1);
                    let p2 = get_u8(&snapshot, w as i32, h as i32, x as i32, y as i32 - 1);
                    let p3 = get_u8(&snapshot, w as i32, h as i32, x as i32 + 1, y as i32 - 1);
                    let p4 = get_u8(&snapshot, w as i32, h as i32, x as i32 - 1, y as i32);
                    let p6 = get_u8(&snapshot, w as i32, h as i32, x as i32 + 1, y as i32);
                    let p7 = get_u8(&snapshot, w as i32, h as i32, x as i32 - 1, y as i32 + 1);
                    let p8 = get_u8(&snapshot, w as i32, h as i32, x as i32, y as i32 + 1);
                    let p9 = get_u8(&snapshot, w as i32, h as i32, x as i32 + 1, y as i32 + 1);
                    if p1 == bg || p2 == bg || p3 == bg || p4 == bg || p6 == bg || p7 == bg
                        || p8 == bg || p9 == bg
                    {
                        self.parent.pixels[y * w + x] = v;
                    } else {
                        self.parent.pixels[y * w + x] = bg;
                    }
                }
            }
        }
    }

    /// Skeletonizes binary objects to single-pixel-wide skeletons (in-place).
    /// Uses two lookup tables and repeated thinning passes.
    pub fn skeletonize(&mut self) {
        let fg = self.foreground;
        let bg = 255 - fg;
        let mut pass = 0;
        loop {
            let removed1 = self.thin(pass, &TABLE, bg);
            pass += 1;
            let removed2 = self.thin(pass, &TABLE, bg);
            pass += 1;
            if removed1 + removed2 == 0 {
                break;
            }
        }
        loop {
            let removed1 = self.thin(pass, &TABLE2, bg);
            pass += 1;
            let removed2 = self.thin(pass, &TABLE2, bg);
            pass += 1;
            if removed1 + removed2 == 0 {
                break;
            }
        }
    }

    /// One thinning pass using the given lookup table. Returns number of pixels removed.
    /// `pass` parity determines which codes are applied.
    fn thin(&mut self, pass: i32, table: &[u8; 256], bg: u8) -> usize {
        let (w, h) = (self.parent.width, self.parent.height);
        let snapshot = self.parent.pixels.clone();
        let mut removed = 0;
        for y in 0..h {
            for x in 0..w {
                let v = snapshot[y * w + x];
                if v != bg {
                    let p1 = get_u8(&snapshot, w as i32, h as i32, x as i32 - 1, y as i32 - 1);
                    let p2 = get_u8(&snapshot, w as i32, h as i32, x as i32, y as i32 - 1);
                    let p3 = get_u8(&snapshot, w as i32, h as i32, x as i32 + 1, y as i32 - 1);
                    let p4 = get_u8(&snapshot, w as i32, h as i32, x as i32 - 1, y as i32);
                    let p6 = get_u8(&snapshot, w as i32, h as i32, x as i32 + 1, y as i32);
                    let p7 = get_u8(&snapshot, w as i32, h as i32, x as i32 - 1, y as i32 + 1);
                    let p8 = get_u8(&snapshot, w as i32, h as i32, x as i32, y as i32 + 1);
                    let p9 = get_u8(&snapshot, w as i32, h as i32, x as i32 + 1, y as i32 + 1);
                    let mut index = 0u32;
                    if p1 != bg {
                        index |= 1;
                    }
                    if p2 != bg {
                        index |= 2;
                    }
                    if p3 != bg {
                        index |= 4;
                    }
                    if p6 != bg {
                        index |= 8;
                    }
                    if p9 != bg {
                        index |= 16;
                    }
                    if p8 != bg {
                        index |= 32;
                    }
                    if p7 != bg {
                        index |= 64;
                    }
                    if p4 != bg {
                        index |= 128;
                    }
                    let code = table[index as usize];
                    if (pass & 1) == 1 {
                        if code == 2 || code == 3 {
                            self.parent.pixels[y * w + x] = bg;
                            removed += 1;
                        }
                    } else {
                        if code == 1 || code == 3 {
                            self.parent.pixels[y * w + x] = bg;
                            removed += 1;
                        }
                    }
                }
            }
        }
        removed
    }
}

/// Safe neighbor access (returns 0 outside bounds, treated as background).
fn get_u8(buf: &[u8], w: i32, h: i32, x: i32, y: i32) -> u8 {
    if x < 0 || y < 0 || x >= w || y >= h {
        0
    } else {
        buf[y as usize * w as usize + x as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::ByteProcessor;

    #[test]
    fn outline_removes_interior() {
        // 5x5 block of foreground (255) with a border of background
        let mut bp = ByteProcessor::from_pixels(
            5,
            5,
            vec![
                0, 0, 0, 0, 0, //  y=0
                0, 255, 255, 255, 0, // y=1
                0, 255, 255, 255, 0, // y=2
                0, 255, 255, 255, 0, // y=3
                0, 0, 0, 0, 0, //  y=4
            ],
        );
        let mut bin = BinaryProcessor::new(&mut bp);
        bin.set_foreground(255);
        bin.outline();
        // Only the hollow ring should remain
        assert_eq!(
            bp.pixels,
            vec![
                0, 0, 0, 0, 0,
                0, 255, 255, 255, 0,
                0, 255, 0, 255, 0,
                0, 255, 255, 255, 0,
                0, 0, 0, 0, 0,
            ]
        );
    }

    #[test]
    fn skeletonize_thins_block_to_center_line() {
        // A 7x1 horizontal line should skeletonize to itself (already 1px)
        let mut bp = ByteProcessor::from_pixels(7, 1, vec![0, 255, 255, 255, 255, 255, 0]);
        let mut bin = BinaryProcessor::new(&mut bp);
        bin.set_foreground(255);
        bin.skeletonize();
        // already 1px wide -> unchanged
        assert_eq!(bp.pixels, vec![0, 255, 255, 255, 255, 255, 0]);
    }

    #[test]
    fn skeletonize_reduces_square() {
        // 9x9 solid square -> thin skeleton
        let mut pixels = vec![0u8; 81];
        for y in 1..8 {
            for x in 1..8 {
                pixels[y * 9 + x] = 255;
            }
        }
        let mut bp = ByteProcessor::from_pixels(9, 9, pixels);
        let before = bp.pixels.iter().filter(|&&v| v == 255).count();
        let mut bin = BinaryProcessor::new(&mut bp);
        bin.set_foreground(255);
        bin.skeletonize();
        let after = bp.pixels.iter().filter(|&&v| v == 255).count();
        // skeleton should be much smaller than the original blob
        assert!(after < before);
        assert!(after > 0);
    }
}