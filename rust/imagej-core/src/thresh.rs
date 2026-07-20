//! Port of `ij.process.AutoThresholder` threshold methods.
//!
//! Only the pure histogram -> threshold functions are ported here. The
//! dispatcher (`getThreshold(Method, histogram)`) with the 256-bin trimming
//! and `bilevel` shortcut is also included, but the full 17-method enum is
//! built up incrementally — this file starts with the three most-used and
//! easiest-to-verify methods: `Mean`, `Otsu`, `IJIsoData`.
//!
//! All functions take `&[u32]` (a histogram; bin counts) and return the
//! threshold bin index, matching the Java `int[] histogram` signatures.

/// Auto-threshold methods ported so far. More variants will be appended
/// to this enum as the port progresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Mean,
    Otsu,
    IJIsoData,
}

impl Method {
    /// Parses a method name (case-sensitive, matching Java `Method.name()`).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Mean" => Some(Self::Mean),
            "Otsu" => Some(Self::Otsu),
            "IJ_IsoData" | "IJIsoData" => Some(Self::IJIsoData),
            _ => None,
        }
    }
}

/// Dispatches to the requested method.
///
/// Mirrors `AutoThresholder.getThreshold`: a `bilevel` shortcut returns
/// early for histograms with <=2 non-zero bins, otherwise the selected
/// method runs over the *trimmed* histogram (first..=last non-zero bin),
/// and the result is offset back by `first`.
///
/// `bilevel_subtract_one` matches the Java default `true`.
pub fn get_threshold(method: Method, histogram: &[u32]) -> i32 {
    if histogram.iter().all(|&c| c == 0) {
        return 0;
    }
    let bilevel = bilevel(histogram, true);
    if bilevel >= 0 {
        return bilevel;
    }

    // Mirror Java: only trim when histogram has more than 256 bins.
    // For <=256 bins, the original passes the histogram through unchanged
    // (minbin stays 0, threshold += 0). We replicate that exactly.
    if histogram.len() > 256 {
        // Trim to first..=last non-zero bin.
        let mut minbin = histogram.len();
        let mut maxbin = 0usize;
        for (i, &c) in histogram.iter().enumerate() {
            if c > 0 {
                if i < minbin {
                    minbin = i;
                }
                maxbin = i;
            }
        }
        if minbin == histogram.len() {
            return 0;
        }
        let trimmed: Vec<u32> = histogram[minbin..=maxbin].to_vec();

        let t = match method {
            Method::Mean => mean(&trimmed),
            Method::Otsu => otsu(&trimmed),
            Method::IJIsoData => ij_isodata(&trimmed),
        };
        return t + minbin as i32;
    }

    let t = match method {
        Method::Mean => mean(histogram),
        Method::Otsu => otsu(histogram),
        Method::IJIsoData => ij_isodata(histogram),
    };
    t
}

/// `AutoThresholder.bilevel`: if the histogram has 1 or 2 non-zero bins,
/// return (last non-zero bin) - (1 if `subtract_one`). Otherwise -1.
pub fn bilevel(data: &[u32], subtract_one: bool) -> i32 {
    let mut first = -1i32;
    let mut second = -1i32;
    let mut non_zero = 0i32;
    for (i, &c) in data.iter().enumerate() {
        if c > 0 {
            non_zero += 1;
            if non_zero > 2 {
                return -1;
            }
            if first == -1 {
                first = i as i32;
            } else {
                second = i as i32;
            }
        }
    }
    if non_zero == 1 {
        first - if subtract_one { 1 } else { 0 }
    } else if non_zero == 2 {
        second - if subtract_one { 1 } else { 0 }
    } else {
        -1
    }
}

/// `AutoThresholder.Mean`: floor( Σ i*data[i] / Σ data[i] ).
/// Uses `u64` to mirror Java's `long` arithmetic (no overflow on 256 bins).
pub fn mean(data: &[u32]) -> i32 {
    let mut tot: u64 = 0;
    let mut sum: u64 = 0;
    for (i, &c) in data.iter().enumerate() {
        tot += c as u64;
        sum += (i as u64) * (c as u64);
    }
    if tot == 0 {
        return 0;
    }
    (sum / tot) as i32
}

/// `AutoThresholder.Otsu`: maximizes between-class variance.
///
/// Port of the Java implementation: cumulative normalized histogram `cnh`,
/// running weighted mean `mean`, and `bcv = (total_mean*cnh - mean)^2 /
/// (cnh*(1-cnh))` maximized over all bins.
pub fn otsu(data: &[u32]) -> i32 {
    let n = data.len();
    let num_pixels: u64 = data.iter().map(|&c| c as u64).sum();
    if num_pixels == 0 {
        return 0;
    }
    let term = 1.0 / num_pixels as f64;

    let mut histo = vec![0.0_f64; n];
    for i in 0..n {
        histo[i] = term * data[i] as f64;
    }

    let mut cnh = vec![0.0_f64; n];
    cnh[0] = histo[0];
    for i in 1..n {
        cnh[i] = cnh[i - 1] + histo[i];
    }

    let mut mean = vec![0.0_f64; n];
    mean[0] = 0.0;
    for i in 1..n {
        mean[i] = mean[i - 1] + (i as f64) * histo[i];
    }

    let total_mean = mean[n - 1];

    let mut threshold: i32 = i32::MIN;
    let mut max_bcv = 0.0_f64;
    for i in 0..n {
        let bcv = total_mean * cnh[i] - mean[i];
        let denom = cnh[i] * (1.0 - cnh[i]);
        let bcv = if denom <= 0.0 {
            0.0
        } else {
            (bcv * bcv) / denom
        };
        if max_bcv < bcv {
            max_bcv = bcv;
            threshold = i as i32;
        }
    }
    threshold
}

/// `AutoThresholder.IJIsoData` (the original ImageJ IsoData implementation).
///
/// Temporarily zeroes `data[0]` and `data[maxValue]`, finds the min/max
/// non-zero bins, then iteratively moves a split point until the average
/// of the two sides equals the split. Returns `round(result)`.
pub fn ij_isodata(data: &[u32]) -> i32 {
    let n = data.len();
    let max_value = n - 1;
    let count0 = data[0];
    let count_max = data[max_value];

    let mut d = data.to_vec();
    d[0] = 0; // exclude erased areas
    d[max_value] = 0;

    let mut min = 0;
    while min < max_value && d[min] == 0 {
        min += 1;
    }
    let mut max = max_value;
    while max > 0 && d[max] == 0 {
        max -= 1;
    }
    if min >= max {
        d[0] = count0;
        d[max_value] = count_max;
        return (n / 2) as i32;
    }

    let mut moving_index = min;
    let mut result = 0.0_f64;
    loop {
        let mut sum1 = 0.0_f64;
        let mut sum2 = 0.0_f64;
        for i in min..=moving_index {
            sum1 += (i as f64) * d[i] as f64;
            sum2 += d[i] as f64;
        }
        let mut sum3 = 0.0_f64;
        let mut sum4 = 0.0_f64;
        for i in (moving_index + 1)..=max {
            sum3 += (i as f64) * d[i] as f64;
            sum4 += d[i] as f64;
        }
        result = if sum2 > 0.0 && sum4 > 0.0 {
            (sum1 / sum2 + sum3 / sum4) / 2.0
        } else {
            result
        };
        moving_index += 1;
        if !((moving_index + 1) <= result as usize && moving_index < max - 1) {
            break;
        }
    }

    d[0] = count0;
    d[max_value] = count_max;
    result.round() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_simple() {
        // single pixel of value 9 -> threshold 9
        let mut h = [0u32; 256];
        h[9] = 1;
        assert_eq!(mean(&h), 9);
    }

    #[test]
    fn mean_two_pixels() {
        // values 0 and 10, equal counts -> floor(10/2)=5
        let mut h = [0u32; 256];
        h[0] = 1;
        h[10] = 1;
        assert_eq!(mean(&h), 5);
    }

    #[test]
    fn bilevel_one_bin() {
        let mut h = [0u32; 256];
        h[50] = 3;
        // single non-zero bin -> 50 - 1 = 49
        assert_eq!(bilevel(&h, true), 49);
    }

    #[test]
    fn bilevel_two_bins() {
        let mut h = [0u32; 256];
        h[10] = 1;
        h[20] = 1;
        // second bin 20 - 1 = 19
        assert_eq!(bilevel(&h, true), 19);
    }

    #[test]
    fn bilevel_many_returns_minus_one() {
        let mut h = [0u32; 256];
        h[1] = 1;
        h[2] = 1;
        h[3] = 1;
        assert_eq!(bilevel(&h, true), -1);
    }

    #[test]
    fn otsu_two_clusters_splits_middle() {
        // Strongly bimodal: background cluster 0..99, foreground cluster 150..255.
        // Otsu maximizes between-class variance; for symmetric equal-weight
        // clusters it returns the last background bin (99) — this matches the
        // reference Java implementation exactly.
        let mut h = [0u32; 256];
        for i in 0..100 {
            h[i] = 10; // background cluster
        }
        for i in 150..256 {
            h[i] = 10; // foreground cluster
        }
        assert_eq!(otsu(&h), 99);
    }

    #[test]
    fn ij_isodata_basic() {
        // Two equal clusters at 10 and 200.
        let mut h = [0u32; 256];
        for i in 0..20 {
            h[i] = 5;
        }
        for i in 190..210 {
            h[i] = 5;
        }
        let t = ij_isodata(&h);
        // Split should be between the two clusters (roughly 100-110).
        assert!((90..=120).contains(&t), "ij_isodata={t}");
    }

    #[test]
    fn get_threshold_dispatch_with_trim() {
        // Bilevel shortcut: only bins 10 and 20 populated.
        let mut h = [0u32; 256];
        h[10] = 1;
        h[20] = 1;
        assert_eq!(get_threshold(Method::Mean, &h), 19);
    }
}
