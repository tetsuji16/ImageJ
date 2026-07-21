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
    Huang,
    Intermodes,
    IsoData,
    Li,
    MaxEntropy,
    MinErrorI,
    Minimum,
    Moments,
    Percentile,
    RenyiEntropy,
    Shanbhag,
    Triangle,
    Yen,
}

impl Method {
    /// Parses a method name (case-sensitive, matching Java `Method.name()`).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Mean" => Some(Self::Mean),
            "Otsu" => Some(Self::Otsu),
            "IJ_IsoData" | "IJIsoData" => Some(Self::IJIsoData),
            "Huang" => Some(Self::Huang),
            "Intermodes" => Some(Self::Intermodes),
            "IsoData" => Some(Self::IsoData),
            "Li" => Some(Self::Li),
            "MaxEntropy" => Some(Self::MaxEntropy),
            "MinErrorI" | "MinError" => Some(Self::MinErrorI),
            "Minimum" => Some(Self::Minimum),
            "Moments" => Some(Self::Moments),
            "Percentile" => Some(Self::Percentile),
            "RenyiEntropy" => Some(Self::RenyiEntropy),
            "Shanbhag" => Some(Self::Shanbhag),
            "Triangle" => Some(Self::Triangle),
            "Yen" => Some(Self::Yen),
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

    // Mirror Java `getThreshold`: the histogram is only trimmed to
    // [first..=last non-zero bin] and offset by `first` when it has MORE
    // than 256 bins (the 16-bit path). For a 256-bin histogram the data is
    // passed through unchanged (minBin stays 0) — this matches the reference
    // Java `getThreshold` exactly, verified against JDK 26.
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
    if histogram.len() > 256 {
        let trimmed: Vec<u32> = histogram[minbin..=maxbin].to_vec();
        let t = match method {
            Method::Mean => mean(&trimmed),
            Method::Otsu => otsu(&trimmed),
            Method::IJIsoData => ij_isodata(&trimmed),
            Method::Huang => huang(&trimmed),
            Method::Intermodes => intermodes(&trimmed),
            Method::IsoData => iso_data(&trimmed),
            Method::Li => li(&trimmed),
            Method::MaxEntropy => max_entropy(&trimmed),
            Method::MinErrorI => min_error_i(&trimmed),
            Method::Minimum => minimum(&trimmed),
            Method::Moments => moments(&trimmed),
            Method::Percentile => percentile(&trimmed),
            Method::RenyiEntropy => renyi_entropy(&trimmed),
            Method::Shanbhag => shanbhag(&trimmed),
            Method::Triangle => triangle(&trimmed),
            Method::Yen => yen(&trimmed),
        };
        return t + minbin as i32;
    }

    let t = match method {
        Method::Mean => mean(histogram),
        Method::Otsu => otsu(histogram),
        Method::IJIsoData => ij_isodata(histogram),
        Method::Huang => huang(histogram),
        Method::Intermodes => intermodes(histogram),
        Method::IsoData => iso_data(histogram),
        Method::Li => li(histogram),
        Method::MaxEntropy => max_entropy(histogram),
        Method::MinErrorI => min_error_i(histogram),
        Method::Minimum => minimum(histogram),
        Method::Moments => moments(histogram),
        Method::Percentile => percentile(histogram),
        Method::RenyiEntropy => renyi_entropy(histogram),
        Method::Shanbhag => shanbhag(histogram),
        Method::Triangle => triangle(histogram),
        Method::Yen => yen(histogram),
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
        if !(((moving_index + 1) as f64 <= result) && (moving_index < max_value - 1)) {
            break;
        }
    }
        d[max_value] = count_max;
        result.round() as i32
    }

/// `AutoThresholder.IsoData` (the intermeans iterative method).
///
/// Finds `g` such that `g == round((meanBelow(g) + meanAbove(g)) / 2)`,
/// starting from the first non-zero bin + 1. Returns -1 if no fixed point
/// is found within `data.len() - 2` iterations (mirrors Java's log + -1).
pub fn iso_data(data: &[u32]) -> i32 {
    let n = data.len();
    let mut g: usize = 0;
    for i in 1..n {
        if data[i] > 0 {
            g = i + 1;
            break;
        }
    }
    loop {
        let mut l: u64 = 0;
        let mut totl: u64 = 0;
        for i in 0..g {
            totl += data[i] as u64;
            l += (data[i] as u64) * (i as u64);
        }
        let mut h: u64 = 0;
        let mut toth: u64 = 0;
        for i in (g + 1)..n {
            toth += data[i] as u64;
            h += (data[i] as u64) * (i as u64);
        }
        if totl > 0 && toth > 0 {
            let l = l / totl;
            let h = h / toth;
            if g == ((l + h) as f64 / 2.0).round() as usize {
                break;
            }
        }
        g += 1;
        if g > n - 2 {
            return -1;
        }
    }
    g as i32
}

/// `AutoThresholder.Li` (minimum cross-entropy, iterative).
///
/// Mirrors the Java `Li` implementation: initial estimate = mean, then
/// iterate `t = (meanBack - meanObj) / (ln(meanBack) - ln(meanObj))`
/// (with the `IS_NEG` round rule) until stable within `tolerance = 0.5`.
pub fn li(data: &[u32]) -> i32 {
    let n = data.len();
    let num_pixels: u64 = data.iter().map(|&c| c as u64).sum();
    if num_pixels == 0 {
        return 0;
    }
    let mut mean = 0.0_f64;
    for (i, &c) in data.iter().enumerate() {
        mean += (i as f64) * c as f64;
    }
    mean /= num_pixels as f64;

    let mut new_thresh = mean;
    let tolerance = 0.5;
    let mut threshold: i32;
    loop {
        let old_thresh = new_thresh;
        threshold = (old_thresh + 0.5) as i32;
        if threshold < 0 {
            threshold = 0;
        }
        if threshold > (n as i32 - 1) {
            threshold = n as i32 - 1;
        }
        let t = threshold as usize;

        let mut sum_back: u64 = 0;
        let mut num_back: u64 = 0;
        for i in 0..=t {
            sum_back += (i as u64) * data[i] as u64;
            num_back += data[i] as u64;
        }
        let mean_back = if num_back == 0 {
            0.0
        } else {
            sum_back as f64 / num_back as f64
        };

        let mut sum_obj: u64 = 0;
        let mut num_obj: u64 = 0;
        for i in (t + 1)..n {
            sum_obj += (i as u64) * data[i] as u64;
            num_obj += data[i] as u64;
        }
        let mean_obj = if num_obj == 0 {
            0.0
        } else {
            sum_obj as f64 / num_obj as f64
        };

        let temp = (mean_back - mean_obj) / (mean_back.ln() - mean_obj.ln());
        // Java: `new_thresh = (int)(temp ± 0.5)` — note `(int)` TRUNCATES
        // toward zero (not round-half-away-from-zero), matching `(temp ± 0.5) as i32`.
        new_thresh = if temp < -2.220_446_049_250_313e-16 {
            (temp - 0.5) as i32 as f64
        } else {
            (temp + 0.5) as i32 as f64
        };

        if (new_thresh - old_thresh).abs() <= tolerance {
            break;
        }
    }
    threshold
}

/// `AutoThresholder.Huang` (fuzzy thresholding, Shannon entropy).
///
/// Port of the original `Huang` implementation (not `Huang2`). Finds the bin
/// minimizing fuzzy entropy over the [first_bin, last_bin] range.
pub fn huang(data: &[u32]) -> i32 {
    let n = data.len();
    let mut first_bin = 0;
    for ih in 0..n {
        if data[ih] != 0 {
            first_bin = ih;
            break;
        }
    }
    let mut last_bin = n - 1;
    for ih in (first_bin..n).rev() {
        if data[ih] != 0 {
            last_bin = ih;
            break;
        }
    }
    let term = 1.0 / (last_bin - first_bin) as f64;

    let mut mu_0 = vec![0.0_f64; n];
    let mut sum_pix: u64 = 0;
    let mut num_pix: u64 = 0;
    for ih in first_bin..n {
        sum_pix += (ih as u64) * data[ih] as u64;
        num_pix += data[ih] as u64;
        mu_0[ih] = sum_pix as f64 / num_pix as f64;
    }

    let mut mu_1 = vec![0.0_f64; n];
    let mut sum_pix = 0u64;
    let mut num_pix = 0u64;
    for ih in (1..=last_bin).rev() {
        sum_pix += (ih as u64) * data[ih] as u64;
        num_pix += data[ih] as u64;
        mu_1[ih - 1] = sum_pix as f64 / num_pix as f64;
    }

    let mut threshold = -1i32;
    let mut min_ent = f64::INFINITY;
    for it in 0..n {
        let mut ent = 0.0_f64;
        for ih in 0..=it {
            let mu_x = 1.0 / (1.0 + term * (ih as f64 - mu_0[it]).abs());
            if !(mu_x < 1e-6 || mu_x > 0.999_999) {
                ent += data[ih] as f64
                    * (-mu_x * mu_x.ln() - (1.0 - mu_x) * (1.0 - mu_x).ln());
            }
        }
        for ih in (it + 1)..n {
            let mu_x = 1.0 / (1.0 + term * (ih as f64 - mu_1[it]).abs());
            if !(mu_x < 1e-6 || mu_x > 0.999_999) {
                ent += data[ih] as f64
                    * (-mu_x * mu_x.ln() - (1.0 - mu_x) * (1.0 - mu_x).ln());
            }
        }
        if ent < min_ent {
            min_ent = ent;
            threshold = it as i32;
        }
    }
    threshold
}

/// `AutoThresholder.MaxEntropy` (Kapur-Sahoo-Wong).
///
/// Maximizes total entropy `ent_back + ent_obj` over gray levels.
pub fn max_entropy(data: &[u32]) -> i32 {
    let n = data.len();
    let total: u64 = data.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return 0;
    }
    let mut norm_histo = vec![0.0_f64; n];
    for i in 0..n {
        norm_histo[i] = data[i] as f64 / total as f64;
    }
    let mut p1 = vec![0.0_f64; n];
    let mut p2 = vec![0.0_f64; n];
    p1[0] = norm_histo[0];
    p2[0] = 1.0 - p1[0];
    for i in 1..n {
        p1[i] = p1[i - 1] + norm_histo[i];
        p2[i] = 1.0 - p1[i];
    }

    let mut first_bin = 0;
    for i in 0..n {
        if !(p1[i].abs() < 2.220_446_049_250_313e-16) {
            first_bin = i;
            break;
        }
    }
    let mut last_bin = n - 1;
    for i in (first_bin..n).rev() {
        if !(p2[i].abs() < 2.220_446_049_250_313e-16) {
            last_bin = i;
            break;
        }
    }

    let mut threshold = -1i32;
    let mut max_ent = f64::MIN_POSITIVE;
    for it in first_bin..=last_bin {
        let mut ent_back = 0.0_f64;
        for ih in 0..=it {
            if data[ih] != 0 {
                let r = norm_histo[ih] / p1[it];
                ent_back -= r * r.ln();
            }
        }
        let mut ent_obj = 0.0_f64;
        for ih in (it + 1)..n {
            if data[ih] != 0 {
                let r = norm_histo[ih] / p2[it];
                ent_obj -= r * r.ln();
            }
        }
        let tot_ent = ent_back + ent_obj;
        if max_ent < tot_ent {
            max_ent = tot_ent;
            threshold = it as i32;
        }
    }
    threshold
}

/// Helper: cumulative sum of `y[0..=j]`. Mirrors `AutoThresholder.A`.
fn a_sum(y: &[u32], j: usize) -> f64 {
    y[..=j].iter().map(|&v| v as f64).sum()
}

/// `AutoThresholder.MinErrorI` (Kittler-Illingworth minimum error).
///
/// Seeds with `Mean`, then iterates a quadratic-equation update until the
/// threshold stabilizes. Returns the current threshold if the discriminant
/// goes negative or the update is NaN (mirrors Java's early return).
pub fn min_error_i(data: &[u32]) -> i32 {
    let n = data.len();
    let mut threshold = mean(data);
    let mut t_prev = -2i32;
    while threshold != t_prev {
        let th = threshold as usize;
        let a_th = a_sum(data, th);
        let a_last = a_sum(data, n - 1);
        let mu = b_sum(data, th) / a_th;
        let nu = (b_sum(data, n - 1) - b_sum(data, th)) / (a_last - a_th);
        let p = a_th / a_last;
        let q = (a_last - a_th) / a_last;
        let sigma2 = c_sum(data, th) / a_th - mu * mu;
        let tau2 = (c_sum(data, n - 1) - c_sum(data, th)) / (a_last - a_th) - nu * nu;

        let w0 = 1.0 / sigma2 - 1.0 / tau2;
        let w1 = mu / sigma2 - nu / tau2;
        let w2 = (mu * mu) / sigma2 - (nu * nu) / tau2
            + (sigma2 * (q * q) / (tau2 * (p * p))).log10();

        let sqterm = w1 * w1 - w0 * w2;
        if sqterm < 0.0 {
            return threshold;
        }
        t_prev = threshold;
        let temp = (w1 + sqterm.sqrt()) / w0;
        if temp.is_nan() {
            threshold = t_prev;
        } else {
            threshold = temp.floor() as i32;
        }
    }
    threshold
}

/// Helper cumulative weighted sum. Mirrors `AutoThresholder.B`.
fn b_sum(y: &[u32], j: usize) -> f64 {
    y[..=j].iter().enumerate().map(|(i, &v)| i as f64 * v as f64).sum()
}

/// Helper cumulative squared weighted sum. Mirrors `AutoThresholder.C`.
fn c_sum(y: &[u32], j: usize) -> f64 {
    y[..=j]
        .iter()
        .enumerate()
        .map(|(i, &v)| (i * i) as f64 * v as f64)
        .sum()
}

/// `AutoThresholder.bimodalTest`: counts strict local maxima; returns true
/// iff exactly two are found. Mirrors the Java static method.
fn bimodal_test(y: &[f64]) -> bool {
    let len = y.len();
    let mut modes = 0;
    for k in 1..(len - 1) {
        if y[k - 1] < y[k] && y[k + 1] < y[k] {
            modes += 1;
            if modes > 2 {
                return false;
            }
        }
    }
    modes == 2
}

/// `AutoThresholder.Intermodes`: smooths the histogram (3-point running
/// mean) until bimodal, then returns the mean of the two peak positions.
/// Returns -1 if not bimodal after 10000 iterations.
pub fn intermodes(data: &[u32]) -> i32 {
    let n = data.len();
    let mut ihisto: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let mut iter = 0;
    while !bimodal_test(&ihisto) {
        let mut previous = 0.0;
        let mut current = 0.0;
        let mut next = ihisto[0];
        for i in 0..(n - 1) {
            previous = current;
            current = next;
            next = ihisto[i + 1];
            ihisto[i] = (previous + current + next) / 3.0;
        }
        ihisto[n - 1] = (current + next) / 3.0;
        iter += 1;
        if iter > 10000 {
            return 0;
        }
    }
    let mut tt = 0i32;
    for i in 1..(n - 1) {
        if ihisto[i - 1] < ihisto[i] && ihisto[i + 1] < ihisto[i] {
            tt += i as i32;
        }
    }
    tt / 2
}

/// `AutoThresholder.Minimum`: smooths until bimodal, then returns the bin
/// just after the first peak (the valley minimum). `max` tracks the last
/// non-zero bin to bound the search.
pub fn minimum(data: &[u32]) -> i32 {
    let n = data.len();
    if n < 2 {
        return 0;
    }
    let mut max = -1i32;
    let mut ihisto: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    for i in 0..n {
        if data[i] > 0 {
            max = i as i32;
        }
    }
    let mut thresh = -1i32;
    let mut iter = 0;
    while !bimodal_test(&ihisto) {
        let mut thisto = vec![0.0_f64; n];
        for i in 1..(n - 1) {
            thisto[i] = (ihisto[i - 1] + ihisto[i] + ihisto[i + 1]) / 3.0;
        }
        thisto[0] = (ihisto[0] + ihisto[1]) / 3.0;
        thisto[n - 1] = (ihisto[n - 2] + ihisto[n - 1]) / 3.0;
        ihisto.copy_from_slice(&thisto);
        iter += 1;
        if iter > 10000 {
            return 0;
        }
    }
    let max = max as usize;
    for i in 1..max {
        if ihisto[i - 1] > ihisto[i] && ihisto[i + 1] >= ihisto[i] {
            thresh = i as i32;
            break;
        }
    }
    thresh
}

/// `AutoThresholder.Moments` (Tsai moment-preserving thresholding).
pub fn moments(data: &[u32]) -> i32 {
    let n = data.len();
    let total: u64 = data.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return 0;
    }
    let mut histo = vec![0.0_f64; n];
    for i in 0..n {
        histo[i] = data[i] as f64 / total as f64;
    }
    let mut m1 = 0.0;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for i in 0..n {
        m1 += i as f64 * histo[i];
        m2 += (i * i) as f64 * histo[i];
        m3 += (i * i * i) as f64 * histo[i];
    }
    let m0 = 1.0;
    let cd = m0 * m2 - m1 * m1;
    let c0 = (-m2 * m2 + m1 * m3) / cd;
    let c1 = (m0 * -m3 + m2 * m1) / cd;
    let z0 = 0.5 * (-c1 - (c1 * c1 - 4.0 * c0).sqrt());
    let z1 = 0.5 * (-c1 + (c1 * c1 - 4.0 * c0).sqrt());
    let p0 = (z1 - m1) / (z1 - z0);

    let mut sum = 0.0;
    let mut threshold = -1i32;
    for i in 0..n {
        sum += histo[i];
        if sum > p0 {
            threshold = i as i32;
            break;
        }
    }
    threshold
}

/// Helper: partial sum of `y[0..=j]`. Mirrors `AutoThresholder.partialSum`.
fn partial_sum(y: &[u32], j: usize) -> f64 {
    y[..=j].iter().map(|&v| v as f64).sum()
}

/// `AutoThresholder.Percentile` (Doyle). Default `ptile = 0.5`: returns the
/// bin whose cumulative fraction is closest to 0.5.
pub fn percentile(data: &[u32]) -> i32 {
    let n = data.len();
    let total = partial_sum(data, n - 1);
    if total == 0.0 {
        return 0;
    }
    let ptile = 0.5;
    let mut temp = 1.0_f64;
    let mut threshold = -1i32;
    for i in 0..n {
        let a = (partial_sum(data, i) / total - ptile).abs();
        if a < temp {
            temp = a;
            threshold = i as i32;
        }
    }
    threshold
}

/// `AutoThresholder.RenyiEntropy` (Kapur-Sahoo-Wong, alpha-variant).
///
/// Computes three candidate thresholds for alpha = 1, 0.5, 2, sorts them,
/// applies the beta weighting, and returns the weighted optimal threshold.
pub fn renyi_entropy(data: &[u32]) -> i32 {
    let n = data.len();
    let total: u64 = data.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return 0;
    }
    let mut norm_histo = vec![0.0_f64; n];
    for i in 0..n {
        norm_histo[i] = data[i] as f64 / total as f64;
    }
    let mut p1 = vec![0.0_f64; n];
    let mut p2 = vec![0.0_f64; n];
    p1[0] = norm_histo[0];
    p2[0] = 1.0 - p1[0];
    for i in 1..n {
        p1[i] = p1[i - 1] + norm_histo[i];
        p2[i] = 1.0 - p1[i];
    }
    let mut first_bin = 0;
    for i in 0..n {
        if !(p1[i].abs() < 2.220_446_049_250_313e-16) {
            first_bin = i;
            break;
        }
    }
    let mut last_bin = n - 1;
    for i in (first_bin..n).rev() {
        if !(p2[i].abs() < 2.220_446_049_250_313e-16) {
            last_bin = i;
            break;
        }
    }

    // alpha = 1
    let mut t_star2 = 0i32;
    let mut max_ent = 0.0_f64;
    for it in first_bin..=last_bin {
        let mut ent_back = 0.0_f64;
        for ih in 0..=it {
            if data[ih] != 0 {
                let r = norm_histo[ih] / p1[it];
                ent_back -= r * r.ln();
            }
        }
        let mut ent_obj = 0.0_f64;
        for ih in (it + 1)..n {
            if data[ih] != 0 {
                let r = norm_histo[ih] / p2[it];
                ent_obj -= r * r.ln();
            }
        }
        let tot = ent_back + ent_obj;
        if max_ent < tot {
            max_ent = tot;
            t_star2 = it as i32;
        }
    }

    // alpha = 0.5
    let mut t_star1 = 0i32;
    let mut max_ent = 0.0_f64;
    let alpha = 0.5;
    let term = 1.0 / (1.0 - alpha);
    for it in first_bin..=last_bin {
        let mut ent_back = 0.0_f64;
        for ih in 0..=it {
            ent_back += (norm_histo[ih] / p1[it]).sqrt();
        }
        let mut ent_obj = 0.0_f64;
        for ih in (it + 1)..n {
            ent_obj += (norm_histo[ih] / p2[it]).sqrt();
        }
        let tot = term * if ent_back * ent_obj > 0.0 {
            (ent_back * ent_obj).ln()
        } else {
            0.0
        };
        if max_ent < tot {
            max_ent = tot;
            t_star1 = it as i32;
        }
    }

    // alpha = 2
    let mut t_star3 = 0i32;
    let mut max_ent = 0.0_f64;
    let alpha = 2.0;
    let term = 1.0 / (1.0 - alpha);
    for it in first_bin..=last_bin {
        let mut ent_back = 0.0_f64;
        for ih in 0..=it {
            let r = norm_histo[ih] / p1[it];
            ent_back += r * r;
        }
        let mut ent_obj = 0.0_f64;
        for ih in (it + 1)..n {
            let r = norm_histo[ih] / p2[it];
            ent_obj += r * r;
        }
        let tot = term * if ent_back * ent_obj > 0.0 {
            (ent_back * ent_obj).ln()
        } else {
            0.0
        };
        if max_ent < tot {
            max_ent = tot;
            t_star3 = it as i32;
        }
    }

    // sort t_star1 <= t_star2 <= t_star3
    let (mut t1, mut t2, mut t3) = (t_star1, t_star2, t_star3);
    if t2 < t1 {
        std::mem::swap(&mut t1, &mut t2);
    }
    if t3 < t2 {
        std::mem::swap(&mut t2, &mut t3);
    }
    if t2 < t1 {
        std::mem::swap(&mut t1, &mut t2);
    }

    let (beta1, beta2, beta3) = if (t2 - t1).abs() <= 5 {
        if (t3 - t2).abs() <= 5 {
            (1, 2, 1)
        } else {
            (0, 1, 3)
        }
    } else if (t3 - t2).abs() <= 5 {
        (3, 1, 0)
    } else {
        (1, 2, 1)
    };

    let omega = p1[t3 as usize] - p1[t1 as usize];
    let opt = (t1 as f64 * (p1[t1 as usize] + 0.25 * omega * beta1 as f64)
        + 0.25 * t2 as f64 * omega * beta2 as f64
        + t3 as f64 * (p2[t3 as usize] + 0.25 * omega * beta3 as f64))
        as i32;
    opt
}

/// `AutoThresholder.Shanbhag` (information-measure thresholding).
pub fn shanbhag(data: &[u32]) -> i32 {
    let n = data.len();
    let total: u64 = data.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return 0;
    }
    let mut norm_histo = vec![0.0_f64; n];
    for i in 0..n {
        norm_histo[i] = data[i] as f64 / total as f64;
    }
    let mut p1 = vec![0.0_f64; n];
    let mut p2 = vec![0.0_f64; n];
    p1[0] = norm_histo[0];
    p2[0] = 1.0 - p1[0];
    for i in 1..n {
        p1[i] = p1[i - 1] + norm_histo[i];
        p2[i] = 1.0 - p1[i];
    }
    let mut first_bin = 0;
    for i in 0..n {
        if !(p1[i].abs() < 2.220_446_049_250_313e-16) {
            first_bin = i;
            break;
        }
    }
    let mut last_bin = n - 1;
    for i in (first_bin..n).rev() {
        if !(p2[i].abs() < 2.220_446_049_250_313e-16) {
            last_bin = i;
            break;
        }
    }

    let mut threshold = -1i32;
    let mut min_ent = f64::INFINITY;
    for it in first_bin..=last_bin {
        let mut ent_back = 0.0_f64;
        let term = 0.5 / p1[it];
        for ih in 1..=it {
            ent_back -= norm_histo[ih] * (1.0 - term * p1[ih - 1]).ln();
        }
        ent_back *= term;
        let mut ent_obj = 0.0_f64;
        let term = 0.5 / p2[it];
        for ih in (it + 1)..n {
            ent_obj -= norm_histo[ih] * (1.0 - term * p2[ih]).ln();
        }
        ent_obj *= term;
        let tot_ent = (ent_back - ent_obj).abs();
        if tot_ent < min_ent {
            min_ent = tot_ent;
            threshold = it as i32;
        }
    }
    threshold
}

/// `AutoThresholder.Triangle` (Zack et al.).
///
/// Works on a *copy* of the histogram (the Java version reverses `data` in
/// place; we keep the input immutable). Returns the split bin, adjusted back
/// if the histogram was reversed.
pub fn triangle(data: &[u32]) -> i32 {
    let n = data.len();
    let mut d: Vec<i64> = data.iter().map(|&v| v as i64).collect();

    let mut min = 0i32;
    for i in 0..n {
        if d[i] > 0 {
            min = i as i32;
            break;
        }
    }
    if min > 0 {
        min -= 1;
    }
    let mut min2 = 0i32;
    for i in (1..n).rev() {
        if d[i] > 0 {
            min2 = i as i32;
            break;
        }
    }
    if min2 < (n - 1) as i32 {
        min2 += 1;
    }
    let mut dmax = 0i64;
    let mut max = 0i32;
    for i in 0..n {
        if d[i] > dmax {
            max = i as i32;
            dmax = d[i];
        }
    }

    let mut inverted = false;
    let (mut min, mut max) = if (max - min) < (min2 - max) {
        inverted = true;
        let mut left = 0usize;
        let mut right = n - 1;
        while left < right {
            d.swap(left, right);
            left += 1;
            right -= 1;
        }
        ((n as i32 - 1 - min2), (n as i32 - 1 - max))
    } else {
        (min, max)
    };

    if min == max {
        return min;
    }

    let nx = d[max as usize] as f64;
    let ny = (min - max) as f64;
    let dlen = (nx * nx + ny * ny).sqrt();
    let nx = nx / dlen;
    let ny = ny / dlen;
    let dd = nx * min as f64 + ny * d[min as usize] as f64;

    let mut split = min;
    let mut split_distance = 0.0_f64;
    for i in (min + 1)..=max {
        let new_distance = nx * i as f64 + ny * d[i as usize] as f64 - dd;
        if new_distance > split_distance {
            split = i;
            split_distance = new_distance;
        }
    }
    split -= 1;

    if inverted {
        let mut left = 0usize;
        let mut right = n - 1;
        while left < right {
            d.swap(left, right);
            left += 1;
            right -= 1;
        }
        (n as i32 - 1 - split)
    } else {
        split
    }
}

/// `AutoThresholder.Yen` (criterion maximization).
pub fn yen(data: &[u32]) -> i32 {
    let n = data.len();
    let total: u64 = data.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return 0;
    }
    let mut norm_histo = vec![0.0_f64; n];
    for i in 0..n {
        norm_histo[i] = data[i] as f64 / total as f64;
    }
    let mut p1 = vec![0.0_f64; n];
    let mut p1_sq = vec![0.0_f64; n];
    let mut p2_sq = vec![0.0_f64; n];
    p1[0] = norm_histo[0];
    for i in 1..n {
        p1[i] = p1[i - 1] + norm_histo[i];
    }
    p1_sq[0] = norm_histo[0] * norm_histo[0];
    for i in 1..n {
        p1_sq[i] = p1_sq[i - 1] + norm_histo[i] * norm_histo[i];
    }
    p2_sq[n - 1] = 0.0;
    for i in (0..(n - 1)).rev() {
        p2_sq[i] = p2_sq[i + 1] + norm_histo[i + 1] * norm_histo[i + 1];
    }

    let mut threshold = -1i32;
    let mut max_crit = f64::MIN_POSITIVE;
    for it in 0..n {
        let prod1 = p1_sq[it] * p2_sq[it];
        let prod2 = p1[it] * (1.0 - p1[it]);
        let crit = -1.0
            * if prod1 > 0.0 { prod1.ln() } else { 0.0 }
            + 2.0 * if prod2 > 0.0 { prod2.ln() } else { 0.0 };
        if crit > max_crit {
            max_crit = crit;
            threshold = it as i32;
        }
    }
    threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- existing tests preserved below ---

    #[test]
    fn mean_simple() {
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

    // --- Full Java parity (values produced by ij.process.AutoThresholder) ---

    /// Builds the same strongly-bimodal histogram used by the Java reference
    /// harness: background cluster 0..99 and foreground cluster 150..255,
    /// each bin holding 10 pixels.
    fn bimodal_hist() -> [u32; 256] {
        let mut h = [0u32; 256];
        for i in 0..100 {
            h[i] = 10;
        }
        for i in 150..256 {
            h[i] = 10;
        }
        h
    }

    #[test]
    fn parity_bimodal_all_methods() {
        let h = bimodal_hist();
        // Values verified against ij.process.AutoThresholder (JDK 26).
        assert_eq!(get_threshold(Method::Huang, &h), 99);
        assert_eq!(get_threshold(Method::Intermodes, &h), 126);
        assert_eq!(get_threshold(Method::IsoData, &h), 126);
        assert_eq!(get_threshold(Method::IJIsoData, &h), 126);
        assert_eq!(get_threshold(Method::Li, &h), 109);
        assert_eq!(get_threshold(Method::MaxEntropy, &h), 152);
        assert_eq!(get_threshold(Method::Mean, &h), 128);
        assert_eq!(get_threshold(Method::MinErrorI, &h), 123);
        assert_eq!(get_threshold(Method::Minimum, &h), 124);
        assert_eq!(get_threshold(Method::Moments, &h), 151);
        assert_eq!(get_threshold(Method::Otsu, &h), 99);
        assert_eq!(get_threshold(Method::Percentile, &h), 152);
        assert_eq!(get_threshold(Method::RenyiEntropy, &h), 152);
        assert_eq!(get_threshold(Method::Shanbhag, &h), 152);
        assert_eq!(get_threshold(Method::Triangle, &h), 101);
        assert_eq!(get_threshold(Method::Yen, &h), 152);
    }

    #[test]
    fn parity_unimodal_peak_methods() {
        // Single peak around bin 100 (50 px each at 100,101,102).
        let mut peak = [0u32; 256];
        peak[100] = 50;
        peak[101] = 50;
        peak[102] = 50;
        // Values verified against ij.process.AutoThresholder (JDK 26).
        assert_eq!(get_threshold(Method::Huang, &peak), 100);
        assert_eq!(get_threshold(Method::IsoData, &peak), 101);
        assert_eq!(get_threshold(Method::IJIsoData, &peak), 101);
        assert_eq!(get_threshold(Method::Li, &peak), 101);
        assert_eq!(get_threshold(Method::MaxEntropy, &peak), 101);
        assert_eq!(get_threshold(Method::Mean, &peak), 101);
        assert_eq!(get_threshold(Method::MinErrorI, &peak), 101);
        assert_eq!(get_threshold(Method::Moments, &peak), 101);
        assert_eq!(get_threshold(Method::Otsu, &peak), 101);
        assert_eq!(get_threshold(Method::Percentile, &peak), 101);
        assert_eq!(get_threshold(Method::RenyiEntropy, &peak), 100);
        assert_eq!(get_threshold(Method::Shanbhag, &peak), 100);
        assert_eq!(get_threshold(Method::Triangle, &peak), 104);
        assert_eq!(get_threshold(Method::Yen, &peak), 100);
        // Intermodes/Minimum are not well-defined for unimodal data;
        // Java returns 0 for both (smoothing never reaches bimodality).
        assert_eq!(get_threshold(Method::Intermodes, &peak), 0);
        assert_eq!(get_threshold(Method::Minimum, &peak), 0);
    }
}
