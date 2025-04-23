use crate::config::MelNorm;
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1};
use num::Complex;

/// Mel filterbanks, within 1.0e-7 of librosa and identical to whisper GGML model-embedded filters.
pub fn mel(
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    f_min: Option<f64>,
    f_max: Option<f64>,
    htk: bool,
    norm: MelNorm,
) -> Array2<f64> {
    let fftfreqs = fft_frequencies(sr, n_fft);
    let f_min: f64 = f_min.unwrap_or(0.0); // Minimum frequency
    let f_max: f64 = f_max.unwrap_or(sr / 2.0); // Maximum frequency
    let mel_f = mel_frequencies(n_mels + 2, f_min, f_max, htk);

    // calculate the triangular mel filter bank weights for mel-frequency cepstral coefficient (MFCC) computation
    let fdiff = &mel_f.slice(s![1..n_mels + 2]) - &mel_f.slice(s![..n_mels + 1]);
    let ramps = &mel_f.slice(s![..n_mels + 2]).insert_axis(Axis(1)) - &fftfreqs;

    let mut weights = Array2::zeros((n_mels, n_fft / 2 + 1));

    for i in 0..n_mels {
        let lower = -&ramps.row(i) / fdiff[i];
        let upper = &ramps.row(i + 2) / fdiff[i + 1];

        weights
            .row_mut(i)
            .assign(&lower.mapv(|x| x.max(0.0).min(1.0)));

        weights
            .row_mut(i)
            .zip_mut_with(&upper.mapv(|x| x.max(0.0).min(1.0)), |a, &b| {
                *a = (*a).min(b);
            });
    }

    match norm {
        MelNorm::Slaney => {
            // Slaney-style mel is scaled to be approx constant energy per channel
            let enorm = 2.0 / (&mel_f.slice(s![2..n_mels + 2]) - &mel_f.slice(s![..n_mels]));
            weights *= &enorm.insert_axis(Axis(1));
        }
    }

    weights
}

pub fn hz_to_mel(frequency: f64, htk: bool) -> f64 {
    if htk {
        return 2595.0 * (1.0 + frequency / 700.0).log10();
    }

    let f_min: f64 = 0.0;
    let f_sp: f64 = 200.0 / 3.0;
    let min_log_hz: f64 = 1000.0;
    let min_log_mel: f64 = (min_log_hz - f_min) / f_sp;
    let logstep: f64 = (6.4f64).ln() / 27.0;

    if frequency >= min_log_hz {
        min_log_mel + ((frequency / min_log_hz).ln() / logstep)
    } else {
        (frequency - f_min) / f_sp
    }
}

pub fn mel_to_hz(mel: f64, htk: bool) -> f64 {
    if htk {
        return 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0);
    }

    let f_min: f64 = 0.0;
    let f_sp: f64 = 200.0 / 3.0;
    let min_log_hz: f64 = 1000.0;
    let min_log_mel: f64 = (min_log_hz - f_min) / f_sp;
    let logstep: f64 = (6.4f64).ln() / 27.0;

    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

pub fn mels_to_hz(mels: ArrayBase<impl Data<Elem = f64>, Ix1>, htk: bool) -> Array1<f64> {
    mels.mapv(|mel| mel_to_hz(mel, htk))
}

pub fn mel_frequencies(n_mels: usize, fmin: f64, fmax: f64, htk: bool) -> Array1<f64> {
    let min_mel = hz_to_mel(fmin, htk);
    let max_mel = hz_to_mel(fmax, htk);

    let mels = Array1::linspace(min_mel, max_mel, n_mels);
    mels_to_hz(mels, htk)
}

pub fn fft_frequencies(sr: f64, n_fft: usize) -> Array1<f64> {
    let step = sr / n_fft as f64;
    let freqs: Array1<f64> = Array1::from_shape_fn(n_fft / 2 + 1, |i| step * i as f64);
    freqs
}

pub use log_mel_spectrogram as log10_mel_spectrogram;

/// Compute a log-Mel spectrogram exactly as in the original OpenAI Whisper (whisper.py & whisper.cpp):
pub fn log_mel_spectrogram(stft: &Array1<Complex<f64>>, mel_filters: &Array2<f64>) -> Array2<f64> {
    let mut mags: Vec<f64> = stft
        .iter()
        .map(|v| v.norm_sqr())
        .take(stft.len() / 2)
        .collect();
    mags.push(0.0);
    let mat = Array2::from_shape_vec((1, mags.len()), mags).unwrap();
    let epsilon = 1e-10;
    mel_filters
        .dot(&mat.t())
        .mapv(|sum| (sum.max(epsilon)).log10())
}

/// Compute a log-Mel spectrogram using the torchaudio/NeMo convention:
/// This matches NeMo’s AudioToMelSpectrogramPreprocessor default (`log_zero_guard_type="add"`,  
/// `log_zero_guard_value=2**-24`, natural log).
pub fn ln_mel_spectrogram(stft: &Array1<Complex<f64>>, mel_filters: &Array2<f64>) -> Array2<f64> {
    let mut mags: Vec<f64> = stft
        .iter()
        .map(|v| v.norm_sqr())
        .take(stft.len() / 2)
        .collect();
    mags.push(0.0);
    let mat = Array2::from_shape_vec((1, mags.len()), mags).unwrap();
    let zero_guard = 2f64.powi(-24);
    mel_filters.dot(&mat.t()).mapv(|x| (x + zero_guard).ln())
}

/// Perform “per_feature” (per-Mel-band) normalization exactly as torchaudio/NeMo does:
/// This is equivalent to PyTorch’s `F.layer_norm` or NeMo’s `normalize="per_feature"`.
pub fn normalize_per_feature(mut mel: Array2<f64>) -> Array2<f64> {
    let zero_guard = 2f64.powi(-24);
    let eps = 1e-5;
    let t = mel.shape()[1] as f64;

    for mut row in mel.axis_iter_mut(Axis(0)) {
        let mean = row.sum() / t;
        let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (t - 1.0);
        let std = (var.max(zero_guard)).sqrt();
        row.mapv_inplace(|v| (v - mean) / (std + eps));
    }

    mel
}

/// Implements the “clamp-and-scale” post-processing first introduced in the
/// original OpenAI Whisper reference (whisper.py) and mirrored in whisper.cpp:
pub fn norm_mel(mel_spec: &Array2<f64>) -> Array2<f64> {
    let mmax = mel_spec.fold(f64::NEG_INFINITY, |acc, x| acc.max(*x));
    let mmax = mmax - 8.0;
    let clamped: Array2<f64> = mel_spec.mapv(|x| (x.max(mmax) + 4.0) / 4.0);

    clamped
}

pub fn norm_mel_vec(mel_spec: &[f32]) -> Vec<f32> {
    let mmax = mel_spec
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let mmax = mmax - 8.0;
    let clamped: Vec<f32> = mel_spec
        .iter()
        .map(|&x| ((x.max(mmax) + 4.0) / 4.0) as f32)
        .collect();

    clamped
}

/// Interleave a mel spectrogram
///
/// Required for creating images or passing to `whisper.cpp`
///
/// Major column order: for waterfall representations where each row is a single time frame.
/// Major row order: interleaved such that each row represents a different frequency band,
/// and each column represents a time step.
///
/// The default is *major row order* - whisper.cpp expects this.
pub fn interleave_frames(
    frames: &[Array2<f64>],
    major_column_order: bool,
    min_width: usize,
) -> Vec<f32> {
    let mut num_frames = frames.len();

    assert!(num_frames > 0, "frames is empty");
    assert!(min_width % 2 == 0, "min_width must be even");

    let num_filters = frames[0].shape()[0];

    let mut frames = frames.to_vec();

    // Ensure an even number of frames by padding with a zeroed frame if necessary
    // *important* mel spectrograms must have even number of columns, otherwise
    // whisper model will give random results.
    if min_width > 0 && num_frames % 2 != 0 {
        frames.push(Array2::from_shape_fn((num_filters, 1), |(_, _)| 0.0));
        num_frames += 1;
    }

    // Calculate the combined width along Axis(1) of all frames
    let combined_width: usize = frames.iter().map(|frame| frame.shape()[1]).sum();

    // Determine the required padding
    let padding = min_width.saturating_sub(combined_width);

    // Create a new Array2 with the required padding
    let padded_frame = Array2::from_shape_fn((num_filters, padding), |(_, _)| 0.0);

    // Insert the padded frame to the end of the frames array if padding is needed
    let mut frames_with_padding = frames.to_vec();
    if padding > 0 {
        frames_with_padding.push(padded_frame);
        num_frames += 1;
    }

    let mut interleaved_data = Vec::with_capacity(num_frames * num_filters * padding);

    if major_column_order {
        for frame_idx in 0..num_frames {
            for filter_idx in 0..num_filters {
                let frame_view = ArrayView2::from(&frames_with_padding[frame_idx]);
                let frame_width = frame_view.shape()[1];
                for x in 0..frame_width {
                    interleaved_data.push(*frame_view.get((filter_idx, x)).unwrap() as f32);
                }
            }
        }
    } else {
        // Interleave in major row order
        for filter_idx in 0..num_filters {
            for frame_idx in 0..num_frames {
                let frame_view = ArrayView2::from(&frames_with_padding[frame_idx]);
                let frame_width = frame_view.shape()[1];
                for x in 0..frame_width {
                    interleaved_data.push(*frame_view.get((filter_idx, x)).unwrap() as f32);
                }
            }
        }
    }

    interleaved_data
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use ndarray_npy::NpzReader;
    use std::fs::File;

    macro_rules! assert_nearby {
        ($left:expr, $right:expr, $epsilon:expr) => {{
            let (left_val, right_val) = (&$left, &$right);
            assert_eq!(
                left_val.len(),
                right_val.len(),
                "Arrays have different lengths"
            );

            for (l, r) in left_val.iter().zip(right_val.iter()) {
                let diff = (*l - *r).abs();
                assert!(
                    diff <= $epsilon,
                    "Assertion failed: left={}, right={}, epsilon={}",
                    l,
                    r,
                    $epsilon
                );
            }
        }};
    }
    #[test]
    fn test_hz_to_mel() {
        let got = vec![hz_to_mel(60.0, false); 1];
        let want = vec![0.9; 1];
        assert_nearby!(got, want, 0.001);
    }

    #[test]
    fn test_mel_to_hz() {
        assert_eq!(mel_to_hz(3.0, false), 200.0);
    }

    #[test]
    fn test_mels_to_hz() {
        let mels = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let want = Array1::from(vec![66.667, 133.333, 200., 266.667, 333.333]);
        let got = mels_to_hz(mels, false);
        assert_nearby!(got, want, 0.001);
    }

    #[test]
    fn test_mel_frequencies() {
        let n_mels = 40;
        let fmin = 0.0;
        let fmax = 11025.0; // librosa.mel_frequencies default max val

        // taken from librosa.mel_frequencies(n_mels=40) in-line example
        let want = Array1::from(vec![
            0., 85.317, 170.635, 255.952, 341.269, 426.586, 511.904, 597.221, 682.538, 767.855,
            853.173, 938.49, 1024.856, 1119.114, 1222.042, 1334.436, 1457.167, 1591.187, 1737.532,
            1897.337, 2071.84, 2262.393, 2470.47, 2697.686, 2945.799, 3216.731, 3512.582, 3835.643,
            4188.417, 4573.636, 4994.285, 5453.621, 5955.205, 6502.92, 7101.009, 7754.107,
            8467.272, 9246.028, 10096.408, 11025.,
        ]);
        let got = mel_frequencies(n_mels, fmin, fmax, false);
        assert_nearby!(got, want, 0.005);
    }

    #[test]
    fn test_fft_frequencies() {
        let sr = 22050.0;
        let n_fft = 16;

        // librosa.fft_frequencies(sr=22050, n_fft=16)
        let want = Array1::from(vec![
            0., 1378.125, 2756.25, 4134.375, 5512.5, 6890.625, 8268.75, 9646.875, 11025.,
        ]);
        let got = fft_frequencies(sr, n_fft);
        assert_nearby!(got, want, 0.001);
    }

    #[test]
    fn test_mel() {
        // whisper mel filterbank
        let file_path = "./testdata/mel_filters.npz";
        let f = File::open(file_path).unwrap();
        let mut npz = NpzReader::new(f).unwrap();
        let filters: Array2<f32> = npz.by_index(0).unwrap();
        let want: Array2<f64> = filters.mapv(|x| f64::from(x));
        let got = mel(16000.0, 400, 80, None, None, false, MelNorm::Slaney);
        assert_eq!(got.shape(), vec![80, 201]);
        for i in 0..80 {
            assert_nearby!(got.row(i), want.row(i), 1.0e-7);
        }
    }

    #[test]
    fn test_nemo_mel_filters() {
        // Load the raw filter‐bank tensor (which is saved as an f32 array
        // with shape [1, n_mels, n_freq_bins])
        let mut npz = NpzReader::new(File::open("testdata/nemo_mel_filters.npz").unwrap()).unwrap();
        let raw: Array3<f32> = npz.by_name("banks").unwrap();
        // Drop the leading singleton batch dimension → shape [n_mels, n_freq_bins]
        let filters_f32: Array2<f32> = raw.index_axis(Axis(0), 0).to_owned();
        // Convert to f64 for comparison
        let want: Array2<f64> = filters_f32.mapv(f64::from);

        // Compute our mel filterbanks independently
        let got = mel(16000.0, 512, 80, None, None, false, MelNorm::Slaney);

        assert_eq!(got.shape(), want.shape());

        for i in 0..got.nrows() {
            assert_nearby!(got.row(i), want.row(i), 1e-7);
        }
    }

    #[test]
    fn test_spectrogram() {
        let fft_size = 400;
        let sampling_rate = 16000.0;
        let n_mels = 80;
        // synthesize a constant-magnitude FFT input
        let fft_input = Array1::from(vec![Complex::new(1.0, 0.0); fft_size]);
        // build mel filterbank directly
        let filters = mel(
            sampling_rate,
            fft_size,
            n_mels,
            None,
            None,
            false,
            MelNorm::Slaney,
        );
        // compute log-mel spectrogram using ln (NeMo style)
        let mel_spec = ln_mel_spectrogram(&fft_input, &filters);
        // should be [n_mels, 1]
        assert_eq!(mel_spec.shape(), &[n_mels, 1]);
    }
}
