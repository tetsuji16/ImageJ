//! Port of `ij.process.ImageStatistics` / `ByteStatistics` (8-bit).
//!
//! Only the *pure* statistical calculations are ported here — the parts
//! that depend only on a histogram (or pixel slice) and produce numbers.
//! ROI geometry, calibration curves, ellipse fitting and centroid over a
//! `ByteProcessor` are deferred (they need the `ImageProcessor` type, which
//! comes later).
//!
//! Reference Java methods:
//! - `ImageStatistics.getRawStatistics`
//! - `ImageStatistics.calculateStdDev`
//! - `ImageStatistics.getRawMinAndMax`
//! - `ImageStatistics.calculateMedian`
//!
//! All functions are `#[inline]`-free pure `fn`s so they can be unit-tested
//! against hand-computed Java-equivalent outcomes.

/// Computes the 256-bin histogram of an 8-bit image (mirrors
/// `ByteProcessor.getHistogram`). Each `u8` value indexes its own bin.
pub fn histogram_8bit(pixels: &[u8]) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &p in pixels {
        hist[p as usize] += 1;
    }
    hist
}

/// Raw statistics computed from a histogram over the inclusive bin range
/// `[min_threshold, max_threshold]`.
///
/// Port of `ImageStatistics.getRawStatistics` + `calculateStdDev`.
///
/// Fields mirror the Java outputs:
/// - `pixel_count`  <- `pixelCount` / `longPixelCount`
/// - `sum`          <- used internally for `mean`
/// - `mean`         <- `mean` (= `umean`, uncalibrated)
/// - `mode`         <- `mode` (bin index with the highest count)
/// - `max_count`    <- `maxCount`
/// - `std_dev`      <- `stdDev` (sample standard deviation, n-1)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawStats {
    pub pixel_count: u64,
    pub mean: f64,
    pub mode: usize,
    pub max_count: u32,
    pub std_dev: f64,
    /// Inclusive min bin index with a non-zero count (from `getRawMinAndMax`).
    pub min_bin: usize,
    /// Inclusive max bin index with a non-zero count (from `getRawMinAndMax`).
    pub max_bin: usize,
}

/// Computes raw statistics from a 256-bin histogram over `[lo, hi]`.
///
/// `getRawMinAndMax` is folded in: `min_bin`/`max_bin` are the first/last
/// bins (within `[lo, hi]`) whose count is non-zero.
pub fn raw_statistics(hist: &[u32; 256], lo: usize, hi: usize) -> RawStats {
    let mut count: u64 = 0;
    let mut sum = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut max_count: u32 = 0;
    let mut mode: usize = lo;
    let mut min_bin: usize = lo;
    let mut max_bin: usize = lo;
    let mut have_min = false;

    for i in lo..=hi {
        let c = hist[i] as u64;
        if c == 0 {
            continue;
        }
        count += c;
        let v = i as f64;
        sum += v * c as f64;
        sum2 += (v * v) * c as f64;
        if hist[i] > max_count {
            max_count = hist[i];
            mode = i;
        }
        if !have_min {
            min_bin = i;
            have_min = true;
        }
        max_bin = i;
    }

    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    let std_dev = calculate_std_dev(count as f64, sum, sum2);

    RawStats {
        pixel_count: count,
        mean,
        mode,
        max_count,
        std_dev,
        min_bin,
        max_bin,
    }
}

/// Sample standard deviation.
///
/// Port of `ImageStatistics.calculateStdDev`:
/// ```text
/// var = (n*sum2 - sum*sum) / n
/// stdDev = sqrt(var / (n - 1))   (if var > 0, else 0)
/// ```
/// Returns 0.0 when `n <= 0`.
pub fn calculate_std_dev(n: f64, sum: f64, sum2: f64) -> f64 {
    if n > 0.0 {
        let var = (n * sum2 - sum * sum) / n;
        if var > 0.0 {
            (var / (n - 1.0)).sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Median computed from a cumulative histogram walk.
///
/// Port of `ImageStatistics.calculateMedian`: walks bins from `first`
/// accumulating counts until the running sum exceeds `pixel_count / 2`,
/// then returns that bin index (calibration omitted — uncalibrated form).
/// Returns `f64::NAN` when `pixel_count == 0`, `first < 0`, or
/// `last >= hist.len()` (mirrors the Java guard).
pub fn calculate_median(hist: &[u32], first: usize, last: usize, pixel_count: u64) -> f64 {
    if pixel_count == 0 || first > last || last >= hist.len() {
        return f64::NAN;
    }
    let half = pixel_count as f64 / 2.0;
    let mut sum = 0u64;
    let mut i = first as isize - 1;
    loop {
        i += 1;
        if i as usize > last {
            break;
        }
        sum += hist[i as usize] as u64;
        if (sum as f64) > half {
            break;
        }
    }
    i as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_basic() {
        let px: [u8; 6] = [0, 0, 1, 2, 2, 2];
        let h = histogram_8bit(&px);
        assert_eq!(h[0], 2);
        assert_eq!(h[1], 1);
        assert_eq!(h[2], 3);
        assert_eq!(h[3], 0);
    }

    #[test]
    fn raw_statistics_matches_hand_computed() {
        // pixels: 10,10,20,30  -> bins 10(x2),20(x1),30(x1)
        let px: [u8; 4] = [10, 10, 20, 30];
        let h = histogram_8bit(&px);
        let s = raw_statistics(&h, 0, 255);
        assert_eq!(s.pixel_count, 4);
        assert_eq!(s.min_bin, 10);
        assert_eq!(s.max_bin, 30);
        assert_eq!(s.mode, 10); // highest count (2)
        assert_eq!(s.max_count, 2);
        // mean = (10+10+20+30)/4 = 17.5
        assert!((s.mean - 17.5).abs() < 1e-12);
        // std dev (sample, n-1):
        // sum2 = 100+100+400+900 = 1500; sum = 70; n = 4
        // var = (4*1500 - 70*70)/4 = (6000-4900)/4 = 275
        // std = sqrt(275/3) = sqrt(91.666...) ≈ 9.57427
        let expected = (275.0_f64 / 3.0).sqrt();
        assert!((s.std_dev - expected).abs() < 1e-12);
    }

    #[test]
    fn std_dev_zero_when_single_value() {
        // all identical -> var = 0 -> std_dev = 0
        let px: [u8; 3] = [5, 5, 5];
        let h = histogram_8bit(&px);
        let s = raw_statistics(&h, 0, 255);
        assert!((s.std_dev - 0.0).abs() < 1e-12);
        assert_eq!(s.mode, 5);
    }

    #[test]
    fn std_dev_empty() {
        assert_eq!(calculate_std_dev(0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn median_walks_cumulative() {
        // bins: 0(x1),1(x1),2(x2) -> counts: 1,1,2 ; total 4 ; half=2
        // sum after bin0 = 1 (<=2); after bin1 = 2 (<=2); after bin2 = 4 (>2) -> median=2
        let hist = [1u32, 1, 2, 0, 0];
        let m = calculate_median(&hist, 0, 4, 4);
        assert_eq!(m, 2.0);
    }

    #[test]
    fn median_nan_on_empty() {
        let hist = [0u32; 256];
        assert!(calculate_median(&hist, 0, 255, 0).is_nan());
    }
}
