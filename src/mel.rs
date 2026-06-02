#[cfg(feature = "ort-tensor")]
use ort::value::Tensor;

use ndarray::{s, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1};
use num::Complex;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::error::Error;
use std::f32::consts::PI as PI_F32;
use std::fmt;
use std::sync::Arc;
/// MelSpectrogram applies a pre-computed filterbank to an FFT result.
/// Results are identical to whisper.cpp and whisper.py
pub struct MelSpectrogram {
    filters: SparseMelFilterbank,
    mel_buf: Vec<f64>,
}

impl MelSpectrogram {
    pub fn new(fft_size: usize, sampling_rate: f64, n_mels: usize) -> Self {
        let filters = mel(sampling_rate, fft_size, n_mels, None, None, false, true);
        let filters = SparseMelFilterbank::from_dense(&filters);
        let mel_buf = vec![0.0; filters.n_mels()];
        Self { filters, mel_buf }
    }

    pub fn add(&mut self, fft: &Array1<Complex<f64>>) -> Array2<f64> {
        self.filters.project_stft_log10(fft, &mut self.mel_buf);
        let normalized = norm_mel_slice_f64(&self.mel_buf);
        Array2::from_shape_vec((self.filters.n_mels(), 1), normalized)
            .expect("mel output shape should match filterbank")
    }
}

#[derive(Clone, Debug)]
pub struct SparseMelWeight {
    pub bin: usize,
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct SparseMelFilterbank {
    rows: Vec<Vec<SparseMelWeight>>,
    fft_bins: usize,
    non_zero_weights: usize,
}

impl SparseMelFilterbank {
    pub fn from_dense(filters: &Array2<f64>) -> Self {
        let rows = filters
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(bin, value)| {
                        (*value != 0.0).then_some(SparseMelWeight {
                            bin,
                            weight: *value,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let non_zero_weights = rows.iter().map(Vec::len).sum();

        Self {
            rows,
            fft_bins: filters.ncols(),
            non_zero_weights,
        }
    }

    pub fn from_mel(
        sample_rate: f64,
        n_fft: usize,
        n_mels: usize,
        f_min: Option<f64>,
        f_max: Option<f64>,
        htk: bool,
        norm: bool,
    ) -> Self {
        let filters = mel(sample_rate, n_fft, n_mels, f_min, f_max, htk, norm);
        Self::from_dense(&filters)
    }

    pub fn n_mels(&self) -> usize {
        self.rows.len()
    }

    pub fn fft_bins(&self) -> usize {
        self.fft_bins
    }

    pub fn non_zero_weights(&self) -> usize {
        self.non_zero_weights
    }

    pub fn dense_weights(&self) -> usize {
        self.rows.len() * self.fft_bins
    }

    pub fn weights_for_mel(&self, mel_idx: usize) -> &[SparseMelWeight] {
        &self.rows[mel_idx]
    }

    pub fn project_power_f64(&self, power: &[f64], output: &mut [f64]) {
        assert_eq!(
            power.len(),
            self.fft_bins,
            "power spectrum length must match filterbank bins"
        );
        assert_eq!(
            output.len(),
            self.rows.len(),
            "output length must match mel count"
        );

        for (mel_idx, row) in self.rows.iter().enumerate() {
            let mut energy = 0.0_f64;
            for weight in row {
                energy += weight.weight * power[weight.bin];
            }
            output[mel_idx] = energy;
        }
    }

    pub fn project_power_f32(&self, power: &[f32], output: &mut [f32]) {
        assert_eq!(
            power.len(),
            self.fft_bins,
            "power spectrum length must match filterbank bins"
        );
        assert_eq!(
            output.len(),
            self.rows.len(),
            "output length must match mel count"
        );

        for (mel_idx, row) in self.rows.iter().enumerate() {
            let mut energy = 0.0_f32;
            for weight in row {
                energy += weight.weight as f32 * power[weight.bin];
            }
            output[mel_idx] = energy;
        }
    }

    fn project_stft_log10(&self, stft: &Array1<Complex<f64>>, output: &mut [f64]) {
        assert_eq!(
            output.len(),
            self.rows.len(),
            "output length must match mel count"
        );

        let half = stft.len() / 2;
        for (mel_idx, row) in self.rows.iter().enumerate() {
            let mut energy = 0.0_f64;
            for weight in row {
                let power = if weight.bin < half {
                    stft[weight.bin].norm_sqr()
                } else {
                    0.0
                };
                energy += weight.weight * power;
            }
            output[mel_idx] = energy.max(1e-10).log10();
        }
    }
}

#[derive(Clone, Debug)]
pub struct BatchLogMelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub f_min: f64,
    pub f_max: Option<f64>,
    pub htk: bool,
    pub norm: bool,
    pub preemphasis: f32,
    pub center: bool,
    pub log_zero_guard: f32,
    pub pad_to: usize,
    pub normalize_per_feature: bool,
}

impl Default for BatchLogMelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 80,
            f_min: 0.0,
            f_max: None,
            htk: false,
            norm: true,
            preemphasis: 0.0,
            center: true,
            log_zero_guard: f32::EPSILON,
            pad_to: 0,
            normalize_per_feature: false,
        }
    }
}

#[derive(Debug)]
pub enum BatchLogMelError {
    InvalidConfig(&'static str),
    Shape(ndarray::ShapeError),
}

impl fmt::Display for BatchLogMelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid log-mel config: {message}"),
            Self::Shape(error) => write!(f, "failed to shape log-mel features: {error}"),
        }
    }
}

impl Error for BatchLogMelError {}

impl From<ndarray::ShapeError> for BatchLogMelError {
    fn from(error: ndarray::ShapeError) -> Self {
        Self::Shape(error)
    }
}

pub struct BatchLogMelOutput {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

pub struct BatchLogMelSpectrogram {
    config: BatchLogMelConfig,
    filters: SparseMelFilterbank,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    fft_bins: usize,
}

impl BatchLogMelSpectrogram {
    pub fn new(config: BatchLogMelConfig) -> Result<Self, BatchLogMelError> {
        validate_batch_config(&config)?;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(config.n_fft);
        let fft_bins = (config.n_fft / 2) + 1;
        let f_max = config.f_max.unwrap_or(config.sample_rate as f64 / 2.0);
        let filters = SparseMelFilterbank::from_mel(
            config.sample_rate as f64,
            config.n_fft,
            config.n_mels,
            Some(config.f_min),
            Some(f_max),
            config.htk,
            config.norm,
        );

        if filters.fft_bins() != fft_bins || filters.n_mels() != config.n_mels {
            return Err(BatchLogMelError::InvalidConfig(
                "mel filterbank shape does not match FFT and mel settings",
            ));
        }

        let window = centered_hann_window_f32(config.n_fft, config.win_length);

        Ok(Self {
            config,
            filters,
            fft,
            window,
            fft_bins,
        })
    }

    pub fn config(&self) -> &BatchLogMelConfig {
        &self.config
    }

    pub fn filters(&self) -> &SparseMelFilterbank {
        &self.filters
    }

    pub fn scratch(&self) -> BatchLogMelScratch {
        BatchLogMelScratch::new(
            self.config.n_fft,
            self.fft.get_inplace_scratch_len(),
            self.fft_bins,
            self.config.n_mels,
        )
    }

    pub fn compute(&self, samples: &[f32]) -> Result<Array2<f32>, BatchLogMelError> {
        let mut scratch = self.scratch();
        self.compute_with_scratch(samples, &mut scratch)
    }

    pub fn compute_flat(&self, samples: &[f32]) -> Result<BatchLogMelOutput, BatchLogMelError> {
        let mut scratch = self.scratch();
        self.compute_flat_with_scratch(samples, &mut scratch)
    }

    pub fn compute_with_scratch(
        &self,
        samples: &[f32],
        scratch: &mut BatchLogMelScratch,
    ) -> Result<Array2<f32>, BatchLogMelError> {
        let output = self.compute_flat_with_scratch(samples, scratch)?;
        Ok(Array2::from_shape_vec(
            (output.rows, output.cols),
            output.data,
        )?)
    }

    pub fn compute_flat_with_scratch(
        &self,
        samples: &[f32],
        scratch: &mut BatchLogMelScratch,
    ) -> Result<BatchLogMelOutput, BatchLogMelError> {
        if samples.is_empty() {
            return Ok(BatchLogMelOutput {
                data: Vec::new(),
                rows: self.config.n_mels,
                cols: 0,
            });
        }

        let valid_frames = self.num_frames(samples.len());
        let padded_frames = pad_len(valid_frames, self.config.pad_to);
        let mut features = vec![0.0_f32; self.config.n_mels * padded_frames];

        scratch.waveform.clear();
        scratch.waveform.extend_from_slice(samples);
        apply_preemphasis(&mut scratch.waveform, self.config.preemphasis);

        prepare_padded_waveform(
            &scratch.waveform,
            &mut scratch.padded,
            self.config.n_fft,
            self.config.center,
        );

        for frame_idx in 0..valid_frames {
            let start = frame_idx * self.config.hop_length;
            for i in 0..self.config.n_fft {
                let sample = scratch.padded.get(start + i).copied().unwrap_or(0.0);
                scratch.fft_input[i] = Complex32::new(sample * self.window[i], 0.0);
            }

            self.fft
                .process_with_scratch(&mut scratch.fft_input, &mut scratch.fft_scratch);

            for (bin_idx, value) in scratch.fft_input.iter().take(self.fft_bins).enumerate() {
                scratch.power[bin_idx] = value.norm_sqr();
            }

            self.filters
                .project_power_f32(&scratch.power, &mut scratch.mel_energy);
            for mel_idx in 0..self.config.n_mels {
                features[(mel_idx * padded_frames) + frame_idx] =
                    (scratch.mel_energy[mel_idx] + self.config.log_zero_guard).ln();
            }
        }

        if self.config.normalize_per_feature {
            normalize_per_feature(
                &mut features,
                self.config.n_mels,
                valid_frames,
                padded_frames,
            );
        }

        Ok(BatchLogMelOutput {
            data: features,
            rows: self.config.n_mels,
            cols: padded_frames,
        })
    }

    fn num_frames(&self, sample_len: usize) -> usize {
        if self.config.center {
            (sample_len / self.config.hop_length) + 1
        } else if sample_len < self.config.n_fft {
            0
        } else {
            ((sample_len - self.config.n_fft) / self.config.hop_length) + 1
        }
    }
}

pub struct BatchLogMelScratch {
    waveform: Vec<f32>,
    padded: Vec<f32>,
    fft_input: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    power: Vec<f32>,
    mel_energy: Vec<f32>,
}

impl BatchLogMelScratch {
    fn new(n_fft: usize, fft_scratch_len: usize, fft_bins: usize, n_mels: usize) -> Self {
        Self {
            waveform: Vec::new(),
            padded: Vec::new(),
            fft_input: vec![Complex32::new(0.0, 0.0); n_fft],
            fft_scratch: vec![Complex32::new(0.0, 0.0); fft_scratch_len],
            power: vec![0.0; fft_bins],
            mel_energy: vec![0.0; n_mels],
        }
    }
}

#[cfg(feature = "ort-tensor")]
pub fn mel_tensor(frames: &[f32], n_mels: usize) -> (Tensor<f32>, Tensor<i64>) {
    let num_frames = frames.len() / n_mels;

    // (1) audio tensor: shape [1, n_mels, num_frames]
    let audio = Tensor::from_array(([1, n_mels as i64, num_frames as i64], frames.to_vec()))
        .expect("failed to create audio tensor");

    // (2) length tensor: shape [1]
    let lengths = Tensor::from_array(([1_i64], vec![num_frames as i64]))
        .expect("failed to create length tensor");

    (audio, lengths)
}

/// The normalised `Array2` output must be processed with [`interleave_frames`]
/// before sending to whisper.cpp
pub fn log_mel_spectrogram(stft: &Array1<Complex<f64>>, mel_filters: &Array2<f64>) -> Array2<f64> {
    let filters = SparseMelFilterbank::from_dense(mel_filters);
    let mut out = vec![0.0; filters.n_mels()];
    filters.project_stft_log10(stft, &mut out);
    Array2::from_shape_vec((filters.n_mels(), 1), out).unwrap()
}

/// Normalisation based on max value in the sample window.
///
/// It's adequate to normalise ftt window-size sample lengths individually but larger sample
/// sizes may sometimes give better results and these functions allow flexibility in the
/// sample size that's normalised over.
pub fn norm_mel(mel_spec: &Array2<f64>) -> Array2<f64> {
    let mmax = mel_spec.fold(f64::NEG_INFINITY, |acc, x| acc.max(*x));
    let mmax = mmax - 8.0;
    let clamped: Array2<f64> = mel_spec.mapv(|x| (x.max(mmax) + 4.0) / 4.0).mapv(|x| x);

    clamped
}

/// Vector-variant of norm_mel.
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

/// Mel filterbanks, within 1.0e-7 of librosa and identical to whisper GGML model-embedded filters.
pub fn mel(
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    f_min: Option<f64>,
    f_max: Option<f64>,
    htk: bool,
    norm: bool,
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

    if norm {
        // Slaney-style mel is scaled to be approx constant energy per channel
        let enorm = 2.0 / (&mel_f.slice(s![2..n_mels + 2]) - &mel_f.slice(s![..n_mels]));
        weights *= &enorm.insert_axis(Axis(1));
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

fn norm_mel_slice_f64(mel_spec: &[f64]) -> Vec<f64> {
    let mmax = mel_spec
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
        - 8.0;
    mel_spec
        .iter()
        .map(|&x| (x.max(mmax) + 4.0) / 4.0)
        .collect()
}

fn validate_batch_config(config: &BatchLogMelConfig) -> Result<(), BatchLogMelError> {
    if config.sample_rate == 0 {
        return Err(BatchLogMelError::InvalidConfig("sample_rate must be > 0"));
    }
    if config.n_fft == 0 {
        return Err(BatchLogMelError::InvalidConfig("n_fft must be > 0"));
    }
    if config.win_length == 0 {
        return Err(BatchLogMelError::InvalidConfig("win_length must be > 0"));
    }
    if config.win_length > config.n_fft {
        return Err(BatchLogMelError::InvalidConfig(
            "win_length must be <= n_fft",
        ));
    }
    if config.hop_length == 0 {
        return Err(BatchLogMelError::InvalidConfig("hop_length must be > 0"));
    }
    if config.n_mels == 0 {
        return Err(BatchLogMelError::InvalidConfig("n_mels must be > 0"));
    }
    if !config.log_zero_guard.is_finite() || config.log_zero_guard <= 0.0 {
        return Err(BatchLogMelError::InvalidConfig(
            "log_zero_guard must be finite and > 0",
        ));
    }
    Ok(())
}

fn prepare_padded_waveform(waveform: &[f32], padded: &mut Vec<f32>, n_fft: usize, center: bool) {
    padded.clear();
    if center {
        let pad = n_fft / 2;
        padded.resize(waveform.len() + (pad * 2), 0.0);
        padded[pad..pad + waveform.len()].copy_from_slice(waveform);
    } else {
        padded.extend_from_slice(waveform);
    }
}

fn apply_preemphasis(waveform: &mut [f32], coeff: f32) {
    if waveform.is_empty() || coeff == 0.0 {
        return;
    }
    let mut prev = waveform[0];
    for sample in waveform.iter_mut().skip(1) {
        let current = *sample;
        *sample = current - (coeff * prev);
        prev = current;
    }
}

fn centered_hann_window_f32(n_fft: usize, win_length: usize) -> Vec<f32> {
    let mut window = vec![0.0_f32; n_fft];
    if win_length <= 1 {
        return window;
    }
    let offset = (n_fft - win_length) / 2;
    for i in 0..win_length {
        let phase = (2.0 * PI_F32 * i as f32) / (win_length as f32 - 1.0);
        window[offset + i] = 0.5 - (0.5 * phase.cos());
    }
    window
}

fn normalize_per_feature(
    features: &mut [f32],
    n_mels: usize,
    valid_frames: usize,
    padded_frames: usize,
) {
    if valid_frames == 0 {
        return;
    }
    for mel_idx in 0..n_mels {
        let start = mel_idx * padded_frames;
        let row = &mut features[start..start + padded_frames];
        let valid = &row[..valid_frames];
        let mean = valid.iter().sum::<f32>() / valid_frames as f32;
        let denom = (valid_frames as f32 - 1.0).max(1.0);
        let variance = valid
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / denom;
        let std = variance.sqrt() + 1e-5;
        for value in row[..valid_frames].iter_mut() {
            *value = (*value - mean) / std;
        }
    }
}

fn pad_len(len: usize, pad_to: usize) -> usize {
    if pad_to == 0 {
        return len;
    }
    len.div_ceil(pad_to) * pad_to
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
        let got = mel(16000.0, 400, 80, None, None, false, true);
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
        let got = mel(16000.0, 512, 80, None, None, false, true);

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
        let mut stage = MelSpectrogram::new(fft_size, sampling_rate, n_mels);
        // Example input data for the FFT
        let fft_input = Array1::from(vec![Complex::new(1.0, 0.0); fft_size]);
        // Add the FFT data to the MelSpectrogram
        let mel_spec = stage.add(&fft_input);
        // Ensure that the output Mel spectrogram has the correct shape
        assert_eq!(mel_spec.shape(), &[n_mels, 1]);
    }

    #[test]
    fn test_sparse_filterbank_matches_dense_projection() {
        let dense = mel(16000.0, 512, 128, Some(0.0), Some(8000.0), false, true);
        let sparse = SparseMelFilterbank::from_dense(&dense);
        let power = (0..257)
            .map(|idx| ((idx as f64 + 1.0) * 0.001).sin().abs())
            .collect::<Vec<_>>();
        let mut got = vec![0.0_f64; 128];
        sparse.project_power_f64(&power, &mut got);

        for mel_idx in 0..128 {
            let mut want = 0.0_f64;
            for bin_idx in 0..257 {
                want += dense[(mel_idx, bin_idx)] * power[bin_idx];
            }
            assert!(
                (got[mel_idx] - want).abs() <= 1e-12,
                "mel {mel_idx}: got {}, want {}",
                got[mel_idx],
                want
            );
        }

        assert!(sparse.non_zero_weights() < sparse.dense_weights() / 10);
    }

    #[test]
    fn test_sparse_mel_spectrogram_matches_dense_api() {
        let fft_size = 512;
        let sampling_rate = 16000.0;
        let n_mels = 80;
        let fft_input = Array1::from(
            (0..fft_size)
                .map(|idx| {
                    let real = (idx as f64 * 0.01).sin();
                    let imag = (idx as f64 * 0.013).cos();
                    Complex::new(real, imag)
                })
                .collect::<Vec<_>>(),
        );
        let filters = mel(sampling_rate, fft_size, n_mels, None, None, false, true);
        let want = norm_mel(&log_mel_spectrogram(&fft_input, &filters));
        let mut sparse = MelSpectrogram::new(fft_size, sampling_rate, n_mels);
        let got = sparse.add(&fft_input);

        assert_eq!(got.shape(), want.shape());
        for row in 0..n_mels {
            assert!(
                (got[(row, 0)] - want[(row, 0)]).abs() <= 1e-12,
                "row {row}: got {}, want {}",
                got[(row, 0)],
                want[(row, 0)]
            );
        }
    }

    #[test]
    fn test_batch_log_mel_config_produces_feature_major_shape() {
        let config = BatchLogMelConfig {
            n_mels: 128,
            preemphasis: 0.97,
            log_zero_guard: 2.0_f32.powi(-24),
            normalize_per_feature: true,
            ..BatchLogMelConfig::default()
        };
        let frontend = BatchLogMelSpectrogram::new(config).unwrap();
        let mut scratch = frontend.scratch();
        let samples = vec![0.0_f32; 16000];

        let features = frontend
            .compute_with_scratch(&samples, &mut scratch)
            .unwrap();

        assert_eq!(features.shape(), &[128, 101]);
    }
}
