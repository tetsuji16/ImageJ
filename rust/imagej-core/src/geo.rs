//! Geometric pixel-buffer operations shared by the processors and the stack
//! processor. Generic over the pixel type so they work for 8/16/32-bit and
//! ARGB images alike. Mirrors `ImageProcessor.flipHorizontal`,
//! `flipVertical`, `rotate90 Degrees` and `applyTable` (8-bit only via LUT).

/// Flips a `w x h` pixel buffer horizontally (left-right).
pub fn flip_horizontal<T: Copy>(pixels: &[T], w: usize, h: usize) -> Vec<T> {
    let mut out = vec![pixels[0]; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + (w - 1 - x)] = pixels[y * w + x];
        }
    }
    out
}

/// Flips a `w x h` pixel buffer vertically (top-bottom).
pub fn flip_vertical<T: Copy>(pixels: &[T], w: usize, h: usize) -> Vec<T> {
    let mut out = vec![pixels[0]; w * h];
    for y in 0..h {
        let dst = (h - 1 - y) * w;
        for x in 0..w {
            out[dst + x] = pixels[y * w + x];
        }
    }
    out
}

/// Rotates a `w x h` pixel buffer 90° clockwise. Result is `h x w`.
pub fn rotate90_cw<T: Copy>(pixels: &[T], w: usize, h: usize) -> Vec<T> {
    let mut out = vec![pixels[0]; w * h];
    for y in 0..h {
        for x in 0..w {
            // destination: column (w-1-x) becomes new row, row y becomes new col
            out[x * h + (h - 1 - y)] = pixels[y * w + x];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_h() {
        // 0 1
        // 2 3
        let p = flip_horizontal(&[0u8, 1, 2, 3], 2, 2);
        assert_eq!(p, vec![1, 0, 3, 2]);
    }

    #[test]
    fn flip_v() {
        let p = flip_vertical(&[0u8, 1, 2, 3], 2, 2);
        assert_eq!(p, vec![2, 3, 0, 1]);
    }

    #[test]
    fn rotate_cw() {
        // 0 1
        // 2 3
        // cw 90 => 2 0
        //        3 1
        let p = rotate90_cw(&[0u8, 1, 2, 3], 2, 2);
        assert_eq!(p, vec![2, 0, 3, 1]);
    }
}
