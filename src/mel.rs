#[cfg(feature = "ort-tensor")]
use ort::value::Tensor;

use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix1};
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::{Complex, Complex32};
use std::error::Error;
use std::f32::consts::PI as PI_F32;
use std::fmt;
use std::sync::Arc;
/// MelSpectrogram applies a pre-computed filterbank to an FFT result.
/// Results are identical to whisper.cpp and whisper.py
pub struct MelSpectrogram {
    filters: SparseMelFilterbank,
    power_buf: Vec<f64>,
    mel_buf: Vec<f64>,
}

impl MelSpectrogram {
    pub fn new(fft_size: usize, sampling_rate: f64, n_mels: usize) -> Self {
        let filters = mel(sampling_rate, fft_size, n_mels, None, None, false, true);
        let filters = SparseMelFilterbank::from_dense(&filters);
        let power_buf = vec![0.0; filters.fft_bins()];
        let mel_buf = vec![0.0; filters.n_mels()];
        Self {
            filters,
            power_buf,
            mel_buf,
        }
    }

    pub fn add(&mut self, fft: &Array1<Complex<f64>>) -> Array2<f64> {
        // Whisper drops the Nyquist bin: only bins below len / 2 contribute.
        let live_bins = fft.len() / 2;
        match fft.as_slice() {
            Some(spectrum) => self.add_spectrum(spectrum, live_bins),
            None => self.add_spectrum(&fft.to_vec(), live_bins),
        }
    }

    /// Same as [`Self::add`] for a spectrum slice that holds at least
    /// `live_bins` bins, such as the half spectrum of a real FFT.
    pub(crate) fn add_spectrum(
        &mut self,
        spectrum: &[Complex<f64>],
        live_bins: usize,
    ) -> Array2<f64> {
        let normalized = self.normalized_frame(spectrum, live_bins);
        Array2::from_shape_vec((normalized.len(), 1), normalized.to_vec())
            .expect("mel output shape should match filterbank")
    }

    /// Log-mel projection plus whisper normalization into the internal buffer.
    pub(crate) fn normalized_frame(
        &mut self,
        spectrum: &[Complex<f64>],
        live_bins: usize,
    ) -> &[f64] {
        whisper_power_spectrum(spectrum, live_bins, &mut self.power_buf);
        self.filters
            .project_log10_f64(&self.power_buf, &mut self.mel_buf);
        norm_mel_in_place_f64(&mut self.mel_buf);
        &self.mel_buf
    }
}

/// Writes `|X[k]|^2` for the first `live_bins` bins of `spectrum` to `power`
/// and zero to the remaining bins. Whisper uses `live_bins = n_fft / 2`, which
/// drops the Nyquist bin.
pub(crate) fn whisper_power_spectrum(
    spectrum: &[Complex<f64>],
    live_bins: usize,
    power: &mut [f64],
) {
    let live_bins = live_bins.min(power.len());
    for (power, value) in power[..live_bins].iter_mut().zip(&spectrum[..live_bins]) {
        *power = value.norm_sqr();
    }
    power[live_bins..].fill(0.0);
}

#[derive(Clone, Debug)]
pub struct SparseMelWeight {
    pub bin: usize,
    pub weight: f64,
}

/// Contiguous run of filter weights for one mel bin. Mel filters are
/// triangles, so the non-zero weights of a row occupy one range of FFT bins.
#[derive(Clone, Copy, Debug)]
struct MelBand {
    start_bin: usize,
    offset: usize,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct SparseMelFilterbank {
    rows: Vec<Vec<SparseMelWeight>>,
    bands: Vec<MelBand>,
    band_weights_f64: Vec<f64>,
    band_weights_f32: Vec<f32>,
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

        // A band spans the first to the last non-zero weight of a row. Any
        // zero inside the span adds an exact 0.0 to the sum, so the band
        // projection keeps the summation order and result of the sparse rows.
        let mut bands = Vec::with_capacity(rows.len());
        let mut band_weights_f64 = Vec::with_capacity(non_zero_weights);
        for (row, weights) in filters.rows().into_iter().zip(&rows) {
            let (start_bin, len) = match (weights.first(), weights.last()) {
                (Some(first), Some(last)) => (first.bin, last.bin + 1 - first.bin),
                _ => (0, 0),
            };
            bands.push(MelBand {
                start_bin,
                offset: band_weights_f64.len(),
                len,
            });
            band_weights_f64.extend(row.iter().skip(start_bin).take(len));
        }
        let band_weights_f32 = band_weights_f64.iter().map(|w| *w as f32).collect();

        Self {
            rows,
            bands,
            band_weights_f64,
            band_weights_f32,
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

        for (band, energy) in self.bands.iter().zip(output.iter_mut()) {
            *energy = band_dot(
                &self.band_weights_f64[band.offset..band.offset + band.len],
                &power[band.start_bin..band.start_bin + band.len],
            );
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

        for (band, energy) in self.bands.iter().zip(output.iter_mut()) {
            *energy = band_dot(
                &self.band_weights_f32[band.offset..band.offset + band.len],
                &power[band.start_bin..band.start_bin + band.len],
            );
        }
    }

    /// Projects `F` power spectra at once. `power` holds the spectra
    /// interleaved by bin (`power[bin * F + frame]`), and `output` receives
    /// the mel energies interleaved by mel (`output[mel * F + frame]`).
    ///
    /// Each frame uses its own SIMD lane and keeps the sequential summation
    /// order of [`Self::project_power_f32`], so the results are identical.
    pub(crate) fn project_frames_f32<const F: usize>(&self, power: &[f32], output: &mut [f32]) {
        project_frames::<f32, F>(
            &self.bands,
            &self.band_weights_f32,
            self.fft_bins,
            power,
            output,
        );
    }

    /// Whisper log-mel energies: `log10(max(energy, 1e-10))` of
    /// [`Self::project_power_f64`].
    pub(crate) fn project_log10_f64(&self, power: &[f64], output: &mut [f64]) {
        self.project_power_f64(power, output);
        for value in output.iter_mut() {
            *value = value.max(1e-10).log10();
        }
    }

    /// The `f64` form of [`Self::project_frames_f32`], identical to
    /// [`Self::project_power_f64`] for each frame.
    pub(crate) fn project_frames_f64<const F: usize>(&self, power: &[f64], output: &mut [f64]) {
        project_frames::<f64, F>(
            &self.bands,
            &self.band_weights_f64,
            self.fft_bins,
            power,
            output,
        );
    }
}

/// Sequential dot product. The fixed summation order keeps results
/// bit-identical to the dense and sparse reference projections.
#[inline]
fn band_dot<T>(weights: &[T], power: &[T]) -> T
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    weights
        .iter()
        .zip(power)
        .fold(T::default(), |energy, (weight, power)| {
            energy + *weight * *power
        })
}

fn project_frames<T, const F: usize>(
    bands: &[MelBand],
    weights: &[T],
    fft_bins: usize,
    power: &[T],
    output: &mut [T],
) where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    assert_eq!(
        power.len(),
        fft_bins * F,
        "power spectra length must match filterbank bins"
    );
    assert_eq!(
        output.len(),
        bands.len() * F,
        "output length must match mel count"
    );

    for (band, energies) in bands.iter().zip(output.chunks_exact_mut(F)) {
        let band_weights = &weights[band.offset..band.offset + band.len];
        let band_power = &power[band.start_bin * F..(band.start_bin + band.len) * F];
        let mut acc = [T::default(); F];
        for (weight, frames) in band_weights.iter().zip(band_power.chunks_exact(F)) {
            for (acc, power) in acc.iter_mut().zip(frames) {
                *acc = *acc + *weight * *power;
            }
        }
        energies.copy_from_slice(&acc);
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
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    fft_bins: usize,
}

impl BatchLogMelSpectrogram {
    pub fn new(config: BatchLogMelConfig) -> Result<Self, BatchLogMelError> {
        validate_batch_config(&config)?;

        let mut planner = RealFftPlanner::<f32>::new();
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
        let mut scratch = BatchLogMelScratch::default();
        self.fit_scratch(&mut scratch);
        scratch
    }

    /// Sizes the per-frame buffers for this frontend, so that a scratch made
    /// by a frontend with other settings stays usable.
    fn fit_scratch(&self, scratch: &mut BatchLogMelScratch) {
        scratch.fft_input.resize(self.config.n_fft, 0.0);
        scratch
            .fft_output
            .resize(self.fft_bins, Complex32::new(0.0, 0.0));
        scratch
            .fft_scratch
            .resize(self.fft.get_scratch_len(), Complex32::new(0.0, 0.0));
        scratch.power.resize(self.fft_bins * FRAME_LANES, 0.0);
        scratch
            .mel_energy
            .resize(self.config.n_mels * FRAME_LANES, 0.0);
        scratch.mel_sums.resize(self.config.n_mels, 0.0);
    }

    pub fn compute(&self, samples: &[f32]) -> Result<Array2<f32>, BatchLogMelError> {
        let mut scratch = self.scratch();
        self.compute_with_scratch(samples, &mut scratch)
    }

    /// Downmix interleaved PCM to mono before feature extraction.
    pub fn compute_interleaved(
        &self,
        samples: &[f32],
        channels: usize,
    ) -> Result<Array2<f32>, BatchLogMelError> {
        let mono = downmix_interleaved(samples, channels)?;
        self.compute(&mono)
    }

    pub fn compute_flat(&self, samples: &[f32]) -> Result<BatchLogMelOutput, BatchLogMelError> {
        let mut scratch = self.scratch();
        self.compute_flat_with_scratch(samples, &mut scratch)
    }

    /// Downmix interleaved PCM to mono before flat feature extraction.
    pub fn compute_flat_interleaved(
        &self,
        samples: &[f32],
        channels: usize,
    ) -> Result<BatchLogMelOutput, BatchLogMelError> {
        let mono = downmix_interleaved(samples, channels)?;
        self.compute_flat(&mono)
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
        if valid_frames == 0 {
            return Ok(BatchLogMelOutput {
                data: features,
                rows: self.config.n_mels,
                cols: padded_frames,
            });
        }

        self.fit_scratch(scratch);
        // Every frame reads n_fft samples. Frames that run past the padded
        // waveform read zeros.
        let frames_len = ((valid_frames - 1) * self.config.hop_length) + self.config.n_fft;
        prepare_padded_waveform(
            samples,
            &mut scratch.padded,
            self.config.n_fft,
            self.config.center,
            self.config.preemphasis,
            frames_len,
        );
        scratch.mel_sums.fill(0.0);

        for group_start in (0..valid_frames).step_by(FRAME_LANES) {
            let group_len = FRAME_LANES.min(valid_frames - group_start);
            if group_len < FRAME_LANES {
                // Unused lanes of the last group are projected but not read.
                scratch.power.fill(0.0);
            }

            for lane in 0..group_len {
                let start = (group_start + lane) * self.config.hop_length;
                let frame = &scratch.padded[start..start + self.config.n_fft];
                for ((input, sample), weight) in
                    scratch.fft_input.iter_mut().zip(frame).zip(&self.window)
                {
                    *input = sample * weight;
                }

                self.fft
                    .process_with_scratch(
                        &mut scratch.fft_input,
                        &mut scratch.fft_output,
                        &mut scratch.fft_scratch,
                    )
                    .expect("scratch buffers are sized for the FFT plan");

                for (frames, value) in scratch
                    .power
                    .chunks_exact_mut(FRAME_LANES)
                    .zip(&scratch.fft_output)
                {
                    frames[lane] = value.norm_sqr();
                }
            }

            self.filters
                .project_frames_f32::<FRAME_LANES>(&scratch.power, &mut scratch.mel_energy);
            for (mel_idx, (energies, sum)) in scratch
                .mel_energy
                .chunks_exact(FRAME_LANES)
                .zip(scratch.mel_sums.iter_mut())
                .enumerate()
            {
                let row_start = (mel_idx * padded_frames) + group_start;
                let row = &mut features[row_start..row_start + group_len];
                for (feature, energy) in row.iter_mut().zip(energies) {
                    let value = (energy + self.config.log_zero_guard).ln();
                    *sum += value;
                    *feature = value;
                }
            }
        }

        if self.config.normalize_per_feature {
            normalize_per_feature(
                &mut features,
                &scratch.mel_sums,
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

/// Frames projected together by [`BatchLogMelSpectrogram`]. Each frame takes
/// one SIMD lane of the mel projection.
pub(crate) const FRAME_LANES: usize = 8;

#[derive(Default)]
pub struct BatchLogMelScratch {
    padded: Vec<f32>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    power: Vec<f32>,
    mel_energy: Vec<f32>,
    mel_sums: Vec<f32>,
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
    let mut power = vec![0.0; filters.fft_bins()];
    match stft.as_slice() {
        Some(spectrum) => whisper_power_spectrum(spectrum, spectrum.len() / 2, &mut power),
        None => whisper_power_spectrum(&stft.to_vec(), stft.len() / 2, &mut power),
    }
    let mut out = vec![0.0; filters.n_mels()];
    filters.project_log10_f64(&power, &mut out);
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
        .map(|&x| (x.max(mmax) + 4.0) / 4.0)
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
    assert!(!frames.is_empty(), "frames is empty");
    assert_eq!(min_width % 2, 0, "min_width must be even");

    let num_filters = frames[0].shape()[0];

    // Ensure an even number of frames by padding with a zeroed frame if necessary
    // *important* mel spectrograms must have even number of columns, otherwise
    // whisper model will give random results.
    let odd_padding = usize::from(min_width > 0 && frames.len() % 2 == 1);

    // Calculate the combined width along Axis(1) of all frames
    let combined_width = frames.iter().map(|frame| frame.shape()[1]).sum::<usize>() + odd_padding;

    // Zero columns appended after the frames to reach `min_width`
    let padding = min_width.saturating_sub(combined_width);
    let zero_columns = odd_padding + padding;

    if major_column_order {
        let mut interleaved_data = Vec::with_capacity(num_filters * (combined_width + padding));
        for frame in frames {
            for filter_idx in 0..num_filters {
                interleaved_data.extend(frame.row(filter_idx).iter().map(|value| *value as f32));
            }
        }
        interleaved_data.resize(interleaved_data.len() + (num_filters * zero_columns), 0.0);
        return interleaved_data;
    }

    // Interleave in major row order. Each frame is read once and written to
    // its columns of every filter row; the zero columns stay as allocated.
    let total_width = combined_width - odd_padding + zero_columns;
    let mut interleaved_data = vec![0.0_f32; num_filters * total_width];
    let mut column = 0;
    for frame in frames {
        let width = frame.ncols();
        for filter_idx in 0..num_filters {
            let start = (filter_idx * total_width) + column;
            for (output, value) in interleaved_data[start..start + width]
                .iter_mut()
                .zip(frame.row(filter_idx))
            {
                *output = *value as f32;
            }
        }
        column += width;
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

        weights.row_mut(i).assign(&lower.mapv(unit_ramp));

        weights
            .row_mut(i)
            .zip_mut_with(&upper.mapv(unit_ramp), |a, &b| {
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

/// Limits a filter ramp to `[0, 1]`, with 0.0 for a NaN ramp. `clamp` would
/// keep the NaN.
fn unit_ramp(x: f64) -> f64 {
    if x > 0.0 {
        x.min(1.0)
    } else {
        0.0
    }
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

pub(crate) fn norm_mel_in_place_f64(mel_spec: &mut [f64]) {
    let mmax = mel_spec
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
        - 8.0;
    for x in mel_spec.iter_mut() {
        *x = (x.max(mmax) + 4.0) / 4.0;
    }
}

/// Downmix interleaved PCM channels to mono by averaging each sample frame.
pub fn downmix_interleaved(samples: &[f32], channels: usize) -> Result<Vec<f32>, BatchLogMelError> {
    if channels == 0 {
        return Err(BatchLogMelError::InvalidConfig(
            "channels must be greater than zero",
        ));
    }
    if !samples.chunks_exact(channels).remainder().is_empty() {
        return Err(BatchLogMelError::InvalidConfig(
            "sample count must be divisible by channels",
        ));
    }
    if channels == 1 {
        return Ok(samples.to_vec());
    }

    let scale = 1.0 / channels as f32;
    Ok(samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() * scale)
        .collect())
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

/// Writes the pre-emphasised waveform into `padded`, with `n_fft / 2` zeros
/// before it when `center` is set, and zeros after it up to `min_len`.
fn prepare_padded_waveform(
    waveform: &[f32],
    padded: &mut Vec<f32>,
    n_fft: usize,
    center: bool,
    preemphasis: f32,
    min_len: usize,
) {
    let pad = if center { n_fft / 2 } else { 0 };
    let len = (waveform.len() + (pad * 2)).max(min_len);
    padded.clear();
    padded.resize(len, 0.0);

    let target = &mut padded[pad..pad + waveform.len()];
    if preemphasis == 0.0 {
        target.copy_from_slice(waveform);
        return;
    }
    if let (Some(first), Some(sample)) = (target.first_mut(), waveform.first()) {
        *first = *sample;
    }
    for (output, pair) in target.iter_mut().skip(1).zip(waveform.windows(2)) {
        *output = pair[1] - (preemphasis * pair[0]);
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

/// Mel rows processed together in the variance pass. Each row keeps its own
/// sequential sum, so the grouping adds instruction-level parallelism without
/// a change to the result.
const NORMALIZE_ROW_GROUP: usize = 8;

/// `mel_sums` holds the sum of the valid frames of each row, accumulated in
/// frame order.
fn normalize_per_feature(
    features: &mut [f32],
    mel_sums: &[f32],
    valid_frames: usize,
    padded_frames: usize,
) {
    if valid_frames == 0 {
        return;
    }
    let count = valid_frames as f32;
    let denom = (count - 1.0).max(1.0);
    let group_len = NORMALIZE_ROW_GROUP * padded_frames;
    for (rows, sums) in features
        .chunks_mut(group_len)
        .zip(mel_sums.chunks(NORMALIZE_ROW_GROUP))
    {
        let mut mean = [0.0_f32; NORMALIZE_ROW_GROUP];
        for (mean, sum) in mean.iter_mut().zip(sums) {
            *mean = sum / count;
        }

        let mut variance = [0.0_f32; NORMALIZE_ROW_GROUP];
        if sums.len() == NORMALIZE_ROW_GROUP {
            for frame_idx in 0..valid_frames {
                for row_idx in 0..NORMALIZE_ROW_GROUP {
                    let centered = rows[(row_idx * padded_frames) + frame_idx] - mean[row_idx];
                    variance[row_idx] += centered * centered;
                }
            }
        } else {
            for row_idx in 0..sums.len() {
                let row = &rows[row_idx * padded_frames..][..valid_frames];
                for value in row {
                    let centered = *value - mean[row_idx];
                    variance[row_idx] += centered * centered;
                }
            }
        }

        for (row_idx, row) in rows.chunks_mut(padded_frames).enumerate() {
            let std = (variance[row_idx] / denom).sqrt() + 1e-5;
            for value in row[..valid_frames].iter_mut() {
                *value = (*value - mean[row_idx]) / std;
            }
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
    use crate::test_support::{fuzz_cases, same_f32, same_f64, Rng};
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
        let want: Array2<f64> = filters.mapv(f64::from);
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

    #[test]
    fn downmix_interleaved_averages_each_sample_frame() {
        let mono = downmix_interleaved(&[1.0, -1.0, 3.0, 1.0, -2.0, -4.0], 2).unwrap();
        assert_eq!(mono, vec![0.0, 2.0, -3.0]);
    }

    #[test]
    fn downmix_interleaved_rejects_incomplete_sample_frames() {
        assert!(matches!(
            downmix_interleaved(&[1.0, 2.0, 3.0], 2),
            Err(BatchLogMelError::InvalidConfig(_))
        ));
        assert!(matches!(
            downmix_interleaved(&[], 0),
            Err(BatchLogMelError::InvalidConfig(_))
        ));
    }

    #[test]
    fn interleaved_batch_matches_mono_batch() {
        let frontend = BatchLogMelSpectrogram::new(BatchLogMelConfig::default()).unwrap();
        let mono = (0..1600)
            .map(|idx| (idx as f32 * 0.013).sin())
            .collect::<Vec<_>>();
        let stereo = mono
            .iter()
            .flat_map(|sample| [*sample, *sample])
            .collect::<Vec<_>>();

        let expected = frontend.compute(&mono).unwrap();
        let actual = frontend.compute_interleaved(&stereo, 2).unwrap();
        assert_eq!(actual, expected);
    }

    /// The 0.4.1 sparse projection: non-zero dense weights in bin order.
    fn dense_projection_f64(dense: &Array2<f64>, power: &[f64], output: &mut [f64]) {
        for (row, energy) in dense.rows().into_iter().zip(output.iter_mut()) {
            *energy = row
                .iter()
                .zip(power)
                .filter(|(weight, _)| **weight != 0.0)
                .fold(0.0, |energy, (weight, power)| energy + weight * power);
        }
    }

    fn dense_projection_f32(dense: &Array2<f64>, power: &[f32], output: &mut [f32]) {
        for (row, energy) in dense.rows().into_iter().zip(output.iter_mut()) {
            *energy = row
                .iter()
                .zip(power)
                .filter(|(weight, _)| **weight != 0.0)
                .fold(0.0, |energy, (weight, power)| {
                    energy + (*weight as f32) * power
                });
        }
    }

    /// The 0.4.1 `MelSpectrogram::add` arithmetic for one full spectrum.
    fn reference_whisper_frame(spectrum: &[Complex<f64>], dense: &Array2<f64>) -> Vec<f64> {
        let half = spectrum.len() / 2;
        let power = (0..dense.ncols())
            .map(|bin| {
                if bin < half {
                    spectrum[bin].norm_sqr()
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let mut energy = vec![0.0; dense.nrows()];
        dense_projection_f64(dense, &power, &mut energy);
        let log = energy
            .iter()
            .map(|energy| energy.max(1e-10).log10())
            .collect::<Vec<_>>();
        norm_mel_vec_f64(&log)
    }

    fn norm_mel_vec_f64(values: &[f64]) -> Vec<f64> {
        let mmax = values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)) - 8.0;
        values.iter().map(|&x| (x.max(mmax) + 4.0) / 4.0).collect()
    }

    /// A dense matrix with empty rows, single weights, runs, runs with
    /// interior zeros, scattered weights, and full rows.
    fn random_dense_filterbank(rng: &mut Rng) -> Array2<f64> {
        let rows = rng.range(1, 24);
        let cols = rng.range(1, 300);
        let mut dense = Array2::zeros((rows, cols));
        for mut row in dense.rows_mut() {
            let start = rng.range(0, cols - 1);
            let end = rng.range(start + 1, cols);
            let pattern = rng.range(0, 5);
            for bin in start..end {
                let keep = match pattern {
                    0 => false,
                    1 => bin == start,
                    2 => true,
                    3 => rng.chance(0.7),
                    4 => rng.chance(0.1),
                    _ => true,
                };
                if keep {
                    row[bin] = (rng.unit() * 3.0) - 1.0;
                }
            }
            if pattern == 5 {
                row.fill(rng.unit() + 0.1);
            }
        }
        dense
    }

    #[test]
    fn fuzz_sparse_projection_matches_dense_reference() {
        let mut rng = Rng::new(1);
        for case in 0..fuzz_cases(300) {
            let dense = random_dense_filterbank(&mut rng);
            let sparse = SparseMelFilterbank::from_dense(&dense);
            let (rows, bins) = dense.dim();
            let spectra = (0..FRAME_LANES)
                .map(|_| {
                    (0..bins)
                        .map(|_| {
                            if rng.chance(0.2) {
                                0.0
                            } else {
                                rng.unit() * 1e3
                            }
                        })
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<_>>();

            let mut want64 = vec![vec![0.0; rows]; FRAME_LANES];
            let mut want32 = vec![vec![0.0_f32; rows]; FRAME_LANES];
            for lane in 0..FRAME_LANES {
                dense_projection_f64(&dense, &spectra[lane], &mut want64[lane]);
                let power32 = spectra[lane].iter().map(|v| *v as f32).collect::<Vec<_>>();
                dense_projection_f32(&dense, &power32, &mut want32[lane]);
            }

            let mut got64 = vec![0.0; rows];
            sparse.project_power_f64(&spectra[0], &mut got64);
            assert_eq!(got64, want64[0], "case {case}: project_power_f64");
            let mut got32 = vec![0.0_f32; rows];
            let power32 = spectra[0].iter().map(|v| *v as f32).collect::<Vec<_>>();
            sparse.project_power_f32(&power32, &mut got32);
            assert_eq!(got32, want32[0], "case {case}: project_power_f32");

            let mut power_lanes64 = vec![0.0; bins * FRAME_LANES];
            let mut power_lanes32 = vec![0.0_f32; bins * FRAME_LANES];
            for bin in 0..bins {
                for lane in 0..FRAME_LANES {
                    power_lanes64[(bin * FRAME_LANES) + lane] = spectra[lane][bin];
                    power_lanes32[(bin * FRAME_LANES) + lane] = spectra[lane][bin] as f32;
                }
            }
            let mut lanes64 = vec![0.0; rows * FRAME_LANES];
            let mut lanes32 = vec![0.0_f32; rows * FRAME_LANES];
            sparse.project_frames_f64::<FRAME_LANES>(&power_lanes64, &mut lanes64);
            sparse.project_frames_f32::<FRAME_LANES>(&power_lanes32, &mut lanes32);
            for row in 0..rows {
                for lane in 0..FRAME_LANES {
                    assert_eq!(
                        lanes64[(row * FRAME_LANES) + lane],
                        want64[lane][row],
                        "case {case}: project_frames_f64 row {row} lane {lane}"
                    );
                    assert_eq!(
                        lanes32[(row * FRAME_LANES) + lane],
                        want32[lane][row],
                        "case {case}: project_frames_f32 row {row} lane {lane}"
                    );
                }
            }
        }
    }

    fn random_batch_config(rng: &mut Rng) -> BatchLogMelConfig {
        let sample_rate = [8_000, 16_000, 22_050, 44_100, 48_000][rng.range(0, 4)];
        let n_fft = if rng.chance(0.5) {
            1 << rng.range(4, 10)
        } else {
            rng.range(16, 1_024)
        };
        let f_min = if rng.chance(0.5) {
            0.0
        } else {
            rng.unit() * 300.0
        };
        BatchLogMelConfig {
            sample_rate,
            n_fft,
            win_length: rng.range(1, n_fft),
            hop_length: rng.range(1, n_fft * 2),
            n_mels: rng.range(1, 96),
            f_min,
            f_max: if rng.chance(0.5) {
                None
            } else {
                Some(f_min + 100.0 + (rng.unit() * (sample_rate as f64 / 2.0 - f_min - 100.0)))
            },
            htk: rng.chance(0.3),
            norm: rng.chance(0.7),
            preemphasis: [0.0, 0.97, rng.unit() as f32][rng.range(0, 2)],
            center: rng.chance(0.7),
            log_zero_guard: [f32::EPSILON, 2.0_f32.powi(-24), 1e-3][rng.range(0, 2)],
            pad_to: [0, 1, rng.range(2, 32)][rng.range(0, 2)],
            normalize_per_feature: rng.chance(0.5),
        }
    }

    #[test]
    fn fuzz_batch_log_mel_matches_frame_by_frame_reference() {
        let mut rng = Rng::new(2);
        // One scratch for all cases, so each frontend resizes a scratch made
        // for other settings.
        let mut scratch = BatchLogMelScratch::default();
        for case in 0..fuzz_cases(60) {
            let config = random_batch_config(&mut rng);
            let frontend = BatchLogMelSpectrogram::new(config.clone()).unwrap();
            let len = if rng.chance(0.1) {
                rng.range(0, 3)
            } else {
                rng.range(0, 6_000)
            };
            let samples = rng.signal(len);
            let got = frontend
                .compute_flat_with_scratch(&samples, &mut scratch)
                .unwrap();
            let want = reference_batch_log_mel(&config, &samples);
            assert_eq!(got.rows, config.n_mels, "case {case}");
            assert_eq!(got.data.len(), got.rows * got.cols, "case {case}");
            assert_eq!(
                got.data.len(),
                want.len(),
                "case {case}: {config:?} len {len}"
            );
            for (idx, (got, want)) in got.data.iter().zip(&want).enumerate() {
                assert!(
                    same_f32(*got, *want),
                    "case {case} value {idx}: {got} vs {want}, {config:?} len {len}"
                );
            }
        }
    }

    #[test]
    fn fuzz_whisper_mel_paths_match_reference() {
        let mut rng = Rng::new(3);
        for case in 0..fuzz_cases(40) {
            let fft_size = rng.range(2, 700);
            let hop_size = rng.range(1, fft_size);
            let n_mels = rng.range(1, 100);
            let sampling_rate = [8_000.0, 16_000.0, 44_100.0][rng.range(0, 2)];
            let len = rng.range(0, 5_000);
            let samples = rng.signal(len);
            let dense = mel(sampling_rate, fft_size, n_mels, None, None, false, true);

            let batch = crate::stft::Spectrogram::compute_mel_spectrogram_cpu(
                &samples,
                fft_size,
                hop_size,
                n_mels,
                sampling_rate,
            );
            let spectra = crate::stft::Spectrogram::compute_all_cpu(&samples, fft_size, hop_size);
            assert_eq!(batch.len(), spectra.len(), "case {case}");

            let mut stage = MelSpectrogram::new(fft_size, sampling_rate, n_mels);
            for (frame_idx, (got, spectrum)) in batch.iter().zip(spectra).enumerate() {
                let want = reference_whisper_frame(&spectrum, &dense);
                let streamed = stage.add(&Array1::from_vec(spectrum));
                for (mel_idx, want) in want.iter().enumerate() {
                    let label = format!(
                        "case {case} fft {fft_size} hop {hop_size} mels {n_mels} \
                         frame {frame_idx} mel {mel_idx}"
                    );
                    assert!(same_f64(streamed[(mel_idx, 0)], *want), "{label}");
                    assert!(same_f32(got[mel_idx], *want as f32), "{label}");
                }
            }
        }
    }

    /// The 0.4.1 `interleave_frames`.
    fn reference_interleave_frames(
        frames: &[Array2<f64>],
        major_column_order: bool,
        min_width: usize,
    ) -> Vec<f32> {
        let num_filters = frames[0].shape()[0];
        let mut frames = frames.to_vec();
        if min_width > 0 && frames.len() % 2 == 1 {
            frames.push(Array2::zeros((num_filters, 1)));
        }
        let combined_width: usize = frames.iter().map(|frame| frame.shape()[1]).sum();
        let padding = min_width.saturating_sub(combined_width);
        if padding > 0 {
            frames.push(Array2::zeros((num_filters, padding)));
        }
        let mut out = Vec::new();
        if major_column_order {
            for frame in &frames {
                for filter_idx in 0..num_filters {
                    for x in 0..frame.shape()[1] {
                        out.push(frame[(filter_idx, x)] as f32);
                    }
                }
            }
        } else {
            for filter_idx in 0..num_filters {
                for frame in &frames {
                    for x in 0..frame.shape()[1] {
                        out.push(frame[(filter_idx, x)] as f32);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn fuzz_interleave_frames_matches_reference() {
        let mut rng = Rng::new(4);
        for case in 0..fuzz_cases(500) {
            let height = rng.range(0, 6);
            let frames = (0..rng.range(1, 12))
                .map(|_| {
                    let width = rng.range(0, 4);
                    Array2::from_shape_fn((height, width), |_| rng.unit() * 10.0 - 5.0)
                })
                .collect::<Vec<_>>();
            let min_width = rng.range(0, 20) * 2;
            for major_column_order in [false, true] {
                assert_eq!(
                    interleave_frames(&frames, major_column_order, min_width),
                    reference_interleave_frames(&frames, major_column_order, min_width),
                    "case {case} column order {major_column_order} min_width {min_width}"
                );
            }
        }
    }

    /// Frame-by-frame form of `BatchLogMelSpectrogram::compute_flat`, with the
    /// same real FFT and the arithmetic of the original implementation.
    fn reference_batch_log_mel(config: &BatchLogMelConfig, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        let frontend = BatchLogMelSpectrogram::new(config.clone()).unwrap();
        let dense = mel(
            config.sample_rate as f64,
            config.n_fft,
            config.n_mels,
            Some(config.f_min),
            Some(config.f_max.unwrap_or(config.sample_rate as f64 / 2.0)),
            config.htk,
            config.norm,
        );
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(config.n_fft);
        let window = centered_hann_window_f32(config.n_fft, config.win_length);

        let mut waveform = samples.to_vec();
        if config.preemphasis != 0.0 {
            for idx in (1..waveform.len()).rev() {
                waveform[idx] = samples[idx] - (config.preemphasis * samples[idx - 1]);
            }
        }
        let pad = if config.center { config.n_fft / 2 } else { 0 };
        let mut padded = vec![0.0_f32; pad];
        padded.extend_from_slice(&waveform);
        padded.resize(padded.len() + pad, 0.0);

        let valid_frames = frontend.num_frames(samples.len());
        let padded_frames = pad_len(valid_frames, config.pad_to);
        let mut features = vec![0.0_f32; config.n_mels * padded_frames];
        let mut input = fft.make_input_vec();
        let mut output = fft.make_output_vec();
        let mut power = vec![0.0_f32; output.len()];
        let mut energy = vec![0.0_f32; config.n_mels];
        for frame_idx in 0..valid_frames {
            let start = frame_idx * config.hop_length;
            for (i, input) in input.iter_mut().enumerate() {
                *input = padded.get(start + i).copied().unwrap_or(0.0) * window[i];
            }
            fft.process(&mut input, &mut output).unwrap();
            for (power, value) in power.iter_mut().zip(&output) {
                *power = value.norm_sqr();
            }
            dense_projection_f32(&dense, &power, &mut energy);
            for (mel_idx, energy) in energy.iter().enumerate() {
                features[(mel_idx * padded_frames) + frame_idx] =
                    (energy + config.log_zero_guard).ln();
            }
        }

        if config.normalize_per_feature && valid_frames > 0 {
            for row in features.chunks_mut(padded_frames) {
                let valid = &row[..valid_frames];
                let mean = valid.iter().sum::<f32>() / valid_frames as f32;
                let denom = (valid_frames as f32 - 1.0).max(1.0);
                let variance = valid
                    .iter()
                    .map(|value| (*value - mean) * (*value - mean))
                    .sum::<f32>()
                    / denom;
                let std = variance.sqrt() + 1e-5;
                for value in row[..valid_frames].iter_mut() {
                    *value = (*value - mean) / std;
                }
            }
        }
        features
    }

    #[test]
    fn batch_log_mel_matches_frame_by_frame_reference() {
        let parakeet = BatchLogMelConfig {
            n_mels: 128,
            preemphasis: 0.97,
            log_zero_guard: 2.0_f32.powi(-24),
            normalize_per_feature: true,
            ..BatchLogMelConfig::default()
        };
        let configs = [
            BatchLogMelConfig::default(),
            parakeet.clone(),
            BatchLogMelConfig {
                pad_to: 16,
                ..parakeet.clone()
            },
            BatchLogMelConfig {
                center: false,
                ..parakeet.clone()
            },
            BatchLogMelConfig {
                n_fft: 400,
                n_mels: 80,
                ..parakeet.clone()
            },
            BatchLogMelConfig {
                n_fft: 401,
                n_mels: 64,
                ..parakeet.clone()
            },
            BatchLogMelConfig {
                n_mels: 13,
                ..parakeet
            },
        ];
        let signal = (0..16_007)
            .map(|idx| {
                let t = idx as f32 / 16_000.0;
                (t * 440.0 * std::f32::consts::TAU).sin() * 0.4
                    + ((idx * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.1
            })
            .collect::<Vec<_>>();

        for config in &configs {
            let frontend = BatchLogMelSpectrogram::new(config.clone()).unwrap();
            let mut scratch = frontend.scratch();
            for len in [0, 1, 159, 160, 399, 400, 401, 1_600, 1_601, 16_007] {
                let samples = &signal[..len];
                let got = frontend
                    .compute_flat_with_scratch(samples, &mut scratch)
                    .unwrap();
                let want = reference_batch_log_mel(config, samples);
                assert_eq!(got.rows, config.n_mels);
                assert_eq!(got.data.len(), got.rows * got.cols);
                assert_eq!(
                    got.data, want,
                    "n_fft {} n_mels {} center {} pad_to {} len {len}",
                    config.n_fft, config.n_mels, config.center, config.pad_to
                );
            }
        }
    }

    #[test]
    fn batch_scratch_from_other_settings_is_resized() {
        let small = BatchLogMelSpectrogram::new(BatchLogMelConfig {
            n_fft: 256,
            win_length: 256,
            n_mels: 40,
            ..BatchLogMelConfig::default()
        })
        .unwrap();
        let large = BatchLogMelSpectrogram::new(BatchLogMelConfig {
            n_mels: 128,
            ..BatchLogMelConfig::default()
        })
        .unwrap();
        let samples = (0..4_000)
            .map(|idx| (idx as f32 * 0.03).sin())
            .collect::<Vec<_>>();

        let mut scratch = small.scratch();
        let got = large
            .compute_flat_with_scratch(&samples, &mut scratch)
            .unwrap();
        let want = large.compute_flat(&samples).unwrap();
        assert_eq!(got.data, want.data);
    }

    #[test]
    fn batch_mel_helper_matches_streaming_mel_frames() {
        let samples = (0..16_000 + 77)
            .map(|idx| ((idx as f32) * 0.021).sin() * 0.5 + ((idx as f32) * 0.0013).cos() * 0.2)
            .collect::<Vec<_>>();
        for (fft_size, hop_size, n_mels) in [(400, 160, 80), (512, 128, 128)] {
            let batch = crate::stft::Spectrogram::compute_mel_spectrogram_cpu(
                &samples, fft_size, hop_size, n_mels, 16_000.0,
            );
            let spectra = crate::stft::Spectrogram::compute_all_cpu(&samples, fft_size, hop_size);
            let mut stage = MelSpectrogram::new(fft_size, 16_000.0, n_mels);
            assert_eq!(batch.len(), spectra.len());
            assert_ne!(batch.len() % FRAME_LANES, 0, "last lane group is partial");
            for (got, spectrum) in batch.iter().zip(spectra) {
                let want = stage
                    .add(&Array1::from_vec(spectrum))
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>();
                assert_eq!(got, &want);
            }
        }
    }

    #[test]
    fn interleave_frames_pads_odd_counts_and_min_width() {
        let frames = (0..3)
            .map(|frame| {
                Array2::from_shape_fn((2, frame + 1), |(row, col)| {
                    (frame * 100 + row * 10 + col) as f64
                })
            })
            .collect::<Vec<_>>();

        // Row-major: each filter row lists all frame columns, then zeros.
        assert_eq!(
            interleave_frames(&frames, false, 10),
            vec![
                0.0, 100.0, 101.0, 200.0, 201.0, 202.0, 0.0, 0.0, 0.0, 0.0, //
                10.0, 110.0, 111.0, 210.0, 211.0, 212.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
        // Column-major: frame by frame, each frame row by row, then zeros.
        assert_eq!(
            interleave_frames(&frames, true, 8),
            vec![
                0.0, 10.0, 100.0, 101.0, 110.0, 111.0, 200.0, 201.0, 202.0, 210.0, 211.0, 212.0,
                0.0, 0.0, 0.0, 0.0,
            ]
        );
        assert_eq!(interleave_frames(&frames, false, 0).len(), 12);
    }
}
