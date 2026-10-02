use crate::mel::{norm_mel_in_place_f64, SparseMelFilterbank, FRAME_LANES};
use ndarray::Array1;
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use std::sync::Arc;

pub struct Spectrogram {
    fft: Arc<dyn RealToComplex<f64>>,
    fft_input: Vec<f64>,
    fft_size: usize,
    idx: u64,
    hop_buf: Vec<f64>,
    hop_size: usize,
    scratch_buf: Vec<Complex<f64>>,
    spectrum: Vec<Complex<f64>>,
    window: Vec<f64>,
}

/// Short Time Fast Fourier Transform
/// Nearly identical to whisper.cpp, pytorch, etc, but the caller might be mindful of the
/// first and final frames:
///   a) pass in exact fft-size sample for initial window to avoid automatic zero-padding
///   b) be aware the final frame will be zero-padded if it is < hop size.
///     - neither is necessary unless you are running additional analysis.
impl Spectrogram {
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(fft_size);
        let idx = 0;

        Self {
            fft_input: fft.make_input_vec(),
            scratch_buf: fft.make_scratch_vec(),
            spectrum: fft.make_output_vec(),
            fft,
            fft_size,
            idx,
            hop_buf: vec![0.0; fft_size],
            hop_size,
            window: hann_window(fft_size),
        }
    }

    /// Takes a single channel of audio (non-interleaved, mono, f32).
    /// Returns an FFT frame using overlap-and-save and the configured `hop_size`
    pub fn add(&mut self, frames: &[f32]) -> Option<Array1<Complex<f64>>> {
        self.add_half(frames)?;
        Some(Array1::from_vec(full_spectrum(
            &self.spectrum,
            self.fft_size,
        )))
    }

    /// Same as [`Self::add`], but returns only the `fft_size / 2 + 1` bins of
    /// the real-input spectrum, without allocation.
    pub(crate) fn add_half(&mut self, frames: &[f32]) -> Option<&[Complex<f64>]> {
        let fft_size = self.fft_size;
        let hop_size = self.hop_size;

        let pcm_size = frames.len();
        assert!(pcm_size <= hop_size, "frames must be <= hop_size");

        self.hop_buf.copy_within(hop_size.., 0);
        // zero pad
        let tail = &mut self.hop_buf[(fft_size - hop_size)..];
        for (sample, frame) in tail.iter_mut().zip(frames) {
            *sample = *frame as f64;
        }
        tail[pcm_size..].fill(0.0);

        self.idx = self.idx.wrapping_add(pcm_size as u64);

        if self.idx < fft_size as u64 {
            return None;
        }

        for ((input, sample), weight) in self
            .fft_input
            .iter_mut()
            .zip(&self.hop_buf)
            .zip(&self.window)
        {
            *input = sample * weight;
        }
        self.fft
            .process_with_scratch(
                &mut self.fft_input,
                &mut self.spectrum,
                &mut self.scratch_buf,
            )
            .expect("buffers are sized for the FFT plan");

        Some(&self.spectrum)
    }

    /// Process all samples at once and return FFT frames in natural order.
    pub fn compute_all_cpu(
        samples: &[f32],
        fft_size: usize,
        hop_size: usize,
    ) -> Vec<Vec<Complex<f64>>> {
        let mut frames_out = Vec::with_capacity(num_frames(samples.len(), fft_size, hop_size));
        for_each_frame_spectrum(samples, fft_size, hop_size, |spectrum| {
            frames_out.push(full_spectrum(spectrum, fft_size));
        });
        frames_out
    }

    /// Compute a mel spectrogram using the current CPU path but in a batched API
    /// that matches the GPU backend's shape and framing semantics.
    pub fn compute_mel_spectrogram_cpu(
        samples: &[f32],
        fft_size: usize,
        hop_size: usize,
        n_mels: usize,
        sampling_rate: f64,
    ) -> Vec<Vec<f32>> {
        // Same filterbank and per-frame math as `MelSpectrogram::add`, with
        // FRAME_LANES frames projected together.
        let filters =
            SparseMelFilterbank::from_mel(sampling_rate, fft_size, n_mels, None, None, false, true);
        // Whisper drops the Nyquist bin, so its power stays zero.
        let live_bins = (fft_size / 2).min(filters.fft_bins());
        let mut power = vec![0.0; filters.fft_bins() * FRAME_LANES];
        let mut energies = vec![0.0; n_mels * FRAME_LANES];
        let mut frame = vec![0.0; n_mels];
        let mut pending = 0;
        let mut out = Vec::with_capacity(num_frames(samples.len(), fft_size, hop_size));

        let mut flush = |power: &[f64], pending: usize, out: &mut Vec<Vec<f32>>| {
            filters.project_frames_f64::<FRAME_LANES>(power, &mut energies);
            for lane in 0..pending {
                for (value, energies) in frame.iter_mut().zip(energies.chunks_exact(FRAME_LANES)) {
                    *value = energies[lane].max(1e-10).log10();
                }
                norm_mel_in_place_f64(&mut frame);
                out.push(frame.iter().map(|v| *v as f32).collect());
            }
        };

        for_each_frame_spectrum(samples, fft_size, hop_size, |spectrum| {
            for (frames, value) in power
                .chunks_exact_mut(FRAME_LANES)
                .zip(&spectrum[..live_bins])
            {
                frames[pending] = value.norm_sqr();
            }
            pending += 1;
            if pending == FRAME_LANES {
                flush(&power, pending, &mut out);
                pending = 0;
            }
        });
        if pending > 0 {
            flush(&power, pending, &mut out);
        }

        out
    }
}

fn num_frames(sample_len: usize, fft_size: usize, hop_size: usize) -> usize {
    if sample_len < fft_size {
        0
    } else {
        (sample_len - fft_size) / hop_size + 1
    }
}

/// Windows each full frame and passes its half spectrum to `on_frame`.
fn for_each_frame_spectrum<F>(samples: &[f32], fft_size: usize, hop_size: usize, mut on_frame: F)
where
    F: FnMut(&[Complex<f64>]),
{
    let frames = num_frames(samples.len(), fft_size, hop_size);
    if frames == 0 {
        return;
    }

    let window = hann_window(fft_size);
    let fft = RealFftPlanner::<f64>::new().plan_fft_forward(fft_size);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();

    for frame_idx in 0..frames {
        let start = frame_idx * hop_size;
        for ((input, sample), weight) in input
            .iter_mut()
            .zip(&samples[start..start + fft_size])
            .zip(&window)
        {
            *input = *sample as f64 * weight;
        }
        fft.process_with_scratch(&mut input, &mut spectrum, &mut scratch)
            .expect("buffers are sized for the FFT plan");
        on_frame(&spectrum);
    }
}

/// Expands the half spectrum of a real input to all `fft_size` bins through
/// conjugate symmetry.
fn full_spectrum(half: &[Complex<f64>], fft_size: usize) -> Vec<Complex<f64>> {
    let mut full = Vec::with_capacity(fft_size);
    full.extend_from_slice(half);
    // Bin k above the half spectrum is conj(bin fft_size - k).
    let mirrored = &half[1..=fft_size - half.len()];
    full.extend(mirrored.iter().rev().map(|value| value.conj()));
    full
}

pub(crate) fn hann_window(fft_size: usize) -> Vec<f64> {
    (0..fft_size)
        .map(|i| 0.5 * (1.0 - f64::cos((2.0 * PI * i as f64) / fft_size as f64)))
        .collect()
}

#[cfg(any(feature = "wgpu", feature = "cuda"))]
pub(crate) fn frame_windows(
    samples: &[f32],
    fft_size: usize,
    hop_size: usize,
    window: &[f64],
) -> Vec<Vec<f64>> {
    if samples.len() < fft_size {
        return Vec::new();
    }

    let num_frames = (samples.len() - fft_size) / hop_size + 1;
    let mut frames = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let windowed = (0..fft_size)
            .map(|i| samples[start + i] as f64 * window[i])
            .collect();
        frames.push(windowed);
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{fuzz_cases, Rng};
    use rustfft::FftPlanner;

    #[test]
    fn test_spectrogram_add() {
        let fft_size = 8;
        let hop_size = 4;
        let mut spectrogram = Spectrogram::new(fft_size, hop_size);

        // Test with frames that have size less than hop_size
        let frames: Vec<f32> = vec![1.0, 2.0, 3.0];
        let fft_frame = spectrogram.add(&frames);
        assert!(fft_frame.is_none());

        // Test with frames that have size equal to hop_size
        let frames: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let fft_frame = spectrogram.add(&frames);
        // None as we have added 7 frames and fft size is 8
        assert!(fft_frame.is_none());
        let frames: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let fft_frame = spectrogram.add(&frames);
        assert!(fft_frame.is_some());
    }

    /// The real-input FFT path must return the full complex spectrum of the
    /// windowed frame, for even and odd FFT sizes.
    #[test]
    fn full_spectrum_matches_complex_fft() {
        for (fft_size, hop_size) in [(400, 160), (512, 160), (9, 4)] {
            let samples = (0..fft_size * 3)
                .map(|idx| ((idx as f32) * 0.37).sin() + ((idx as f32) * 0.011).cos())
                .collect::<Vec<_>>();
            let window = hann_window(fft_size);
            let fft = FftPlanner::<f64>::new().plan_fft_forward(fft_size);
            let reference = |start: usize| {
                let mut spectrum = (0..fft_size)
                    .map(|i| Complex::new(samples[start + i] as f64 * window[i], 0.0))
                    .collect::<Vec<_>>();
                fft.process(&mut spectrum);
                spectrum
            };
            let assert_close = |got: &[Complex<f64>], want: &[Complex<f64>], label: &str| {
                assert_eq!(got.len(), fft_size);
                for (bin, (got, want)) in got.iter().zip(want).enumerate() {
                    assert!(
                        (got - want).norm() <= 1e-9,
                        "{label} fft_size {fft_size} bin {bin}: {got} vs {want}"
                    );
                }
            };

            let batch = Spectrogram::compute_all_cpu(&samples, fft_size, hop_size);
            assert!(!batch.is_empty());
            for (frame_idx, got) in batch.iter().enumerate() {
                assert_close(got, &reference(frame_idx * hop_size), "batch");
            }

            // After k hops the streaming buffer holds the last fft_size samples.
            let mut streaming = Spectrogram::new(fft_size, hop_size);
            let mut streamed = 0;
            for (chunk_idx, chunk) in samples.chunks_exact(hop_size).enumerate() {
                if let Some(got) = streaming.add(chunk) {
                    let end = (chunk_idx + 1) * hop_size;
                    assert_close(
                        got.as_slice().unwrap(),
                        &reference(end - fft_size),
                        "stream",
                    );
                    streamed += 1;
                }
            }
            assert!(streamed > 0);
        }
    }

    fn complex_reference(windowed: &[f64]) -> Vec<Complex<f64>> {
        let mut spectrum = windowed
            .iter()
            .map(|value| Complex::new(*value, 0.0))
            .collect::<Vec<_>>();
        FftPlanner::<f64>::new()
            .plan_fft_forward(spectrum.len())
            .process(&mut spectrum);
        spectrum
    }

    fn assert_spectrum_close(got: &[Complex<f64>], want: &[Complex<f64>], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}");
        let scale = 1.0 + want.iter().map(|value| value.norm()).sum::<f64>();
        for (bin, (got, want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).norm() <= 1e-12 * scale,
                "{label} bin {bin}: {got} vs {want}"
            );
        }
    }

    /// Random FFT and hop sizes with chunks shorter than, or equal to, the hop.
    /// The model is the 0.4.1 overlap buffer with a complex FFT.
    #[test]
    fn fuzz_streaming_spectrogram_matches_model() {
        let mut rng = Rng::new(5);
        for case in 0..fuzz_cases(150) {
            let fft_size = rng.range(1, 520);
            let hop_size = rng.range(1, fft_size);
            let window = hann_window(fft_size);
            let mut spectrogram = Spectrogram::new(fft_size, hop_size);
            let mut buffer = vec![0.0_f64; fft_size];
            let mut received = 0_u64;

            for chunk_idx in 0..rng.range(1, 40) {
                let chunk_len = if rng.chance(0.6) {
                    hop_size
                } else {
                    rng.range(0, hop_size)
                };
                let chunk = rng.signal(chunk_len);

                buffer.copy_within(hop_size.., 0);
                let tail = fft_size - hop_size;
                for (idx, value) in buffer[tail..].iter_mut().enumerate() {
                    *value = chunk.get(idx).map_or(0.0, |sample| *sample as f64);
                }
                received += chunk_len as u64;

                let got = spectrogram.add(&chunk);
                let label = format!("case {case} fft {fft_size} hop {hop_size} chunk {chunk_idx}");
                if received < fft_size as u64 {
                    assert!(got.is_none(), "{label}");
                    continue;
                }
                let windowed = buffer
                    .iter()
                    .zip(&window)
                    .map(|(sample, weight)| sample * weight)
                    .collect::<Vec<_>>();
                let got = got.unwrap_or_else(|| panic!("{label}: missing frame"));
                assert_spectrum_close(
                    got.as_slice().unwrap(),
                    &complex_reference(&windowed),
                    &label,
                );
            }
        }
    }

    #[test]
    fn fuzz_compute_all_cpu_matches_complex_fft() {
        let mut rng = Rng::new(6);
        for case in 0..fuzz_cases(100) {
            let fft_size = rng.range(1, 700);
            let hop_size = rng.range(1, fft_size * 2);
            let len = rng.range(0, 4_000);
            let samples = rng.signal(len);
            let window = hann_window(fft_size);
            let frames = Spectrogram::compute_all_cpu(&samples, fft_size, hop_size);
            let expected_frames = if samples.len() < fft_size {
                0
            } else {
                ((samples.len() - fft_size) / hop_size) + 1
            };
            assert_eq!(frames.len(), expected_frames, "case {case}");
            for (frame_idx, got) in frames.iter().enumerate() {
                let start = frame_idx * hop_size;
                let windowed = samples[start..start + fft_size]
                    .iter()
                    .zip(&window)
                    .map(|(sample, weight)| *sample as f64 * weight)
                    .collect::<Vec<_>>();
                let label = format!("case {case} fft {fft_size} hop {hop_size} frame {frame_idx}");
                assert_spectrum_close(got, &complex_reference(&windowed), &label);
            }
        }
    }
}
