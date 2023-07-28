use ndarray::Array1;
use num::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

pub struct Stage {
    complex_buf: Vec<Complex<f64>>,
    fft: Arc<dyn Fft<f64>>,
    fft_size: usize,
    idx: u64,
    hop_buf: Vec<f64>,
    hop_size: usize,
    scratch_buf: Vec<Complex<f64>>,
    window: Vec<f64>,
}

/// stream-based stft processor.
/// Nearly identical to whisper.cpp, pytorch, etc, but the caller might be mindful of the
/// first and final frames:
///   a) pass in exact fft-size sample for initial window to avoid automatic zero-padding
///   b) be aware the final frame will be zero-padded if it is < hop size.
///     - neither is necessary unless you are running additional analysis.
impl Stage {
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - f64::cos((2.0 * PI * i as f64) / fft_size as f64)))
            .collect();
        let mut idx = 0;

        Self {
            complex_buf: vec![Complex::new(0.0, 0.0); fft_size],
            fft,
            fft_size,
            idx,
            hop_buf: vec![0.0; fft_size],
            hop_size,
            scratch_buf: vec![Complex::new(0.0, 0.0); fft_size],
            window,
        }
    }

    /// Non-interleaved PCM frames for a single channel
    /// Returns
    /// ~580 microseconds (fft_size=400) on M2 Air.
    pub fn add(&mut self, frames: &Vec<f32>) -> Option<Array1<Complex<f64>>> {
        let fft_size = self.fft_size;
        let hop_size = self.hop_size;

        let mut pcm_data: Vec<f64> = frames.iter().map(|x| *x as f64).collect();
        let pcm_size = pcm_data.len();
        assert!(pcm_size <= hop_size, "frames must be <= hop_size");

        // zero pad
        if pcm_size < hop_size {
            pcm_data.extend_from_slice(&vec![0.0; hop_size - pcm_size]);
        }

        self.hop_buf.copy_within(hop_size.., 0);
        self.hop_buf[(fft_size - hop_size)..].copy_from_slice(&pcm_data);

        self.idx = self.idx.wrapping_add(pcm_size as u64);

        if self.idx >= fft_size as u64 {
            let windowed_samples: Vec<f64> = self
                .hop_buf
                .iter()
                .enumerate()
                .map(|(j, val)| val * self.window[j])
                .collect();

            self.complex_buf
                .iter_mut()
                .zip(windowed_samples.iter())
                .for_each(|(c, val)| *c = Complex::new(*val, 0.0));

            self.fft
                .process_with_scratch(&mut self.complex_buf, &mut self.scratch_buf);

            Some(Array1::from_vec(self.complex_buf.clone()))
        } else {
            None
        }
    }
}
