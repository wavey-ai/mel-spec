use ndarray::Array1;
use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::f64::consts::PI;
use std::sync::Arc;

pub struct Spectrogram {
    real_buf: Vec<f64>,
    output_buf: Vec<Complex<f64>>,
    r2c: Arc<dyn RealToComplex<f64>>,
    fft_size: usize,
    idx: u64,
    hop_buf: Vec<f64>,
    hop_size: usize,
    scratch_buf: Vec<Complex<f64>>,
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
        let mut planner = RealFftPlanner::new();
        let r2c = planner.plan_fft_forward(fft_size);
        
        // Create output buffer that's the right size for real FFT 
        // (fft_size/2 + 1 complex values)
        let mut output_buf = r2c.make_output_vec();
        let scratch_buf = r2c.make_scratch_vec();
        
        // Hann window
        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - f64::cos((2.0 * PI * i as f64) / fft_size as f64)))
            .collect();
        let idx = 0;

        Self {
            real_buf: vec![0.0; fft_size],
            output_buf,
            r2c,
            fft_size,
            idx,
            hop_buf: vec![0.0; fft_size],
            hop_size,
            scratch_buf,
            window,
        }
    }

    /// Takes a single channel of audio (non-interleaved, mono, f32).
    /// Returns an FFT frame using overlap-and-save and the configured `hop_size`
    pub fn add(&mut self, frames: &[f32]) -> Option<Array1<Complex<f64>>> {
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
            // Apply window directly to real_buf
            for i in 0..fft_size {
                self.real_buf[i] = self.hop_buf[i] * self.window[i];
            }

            // Process with realfft
            self.r2c.process_with_scratch(
                &mut self.real_buf, 
                &mut self.output_buf, 
                &mut self.scratch_buf
            ).unwrap();

            // Convert the output to a format compatible with existing code
            // Since realfft only returns fft_size/2 + 1 complex values, we need to
            // convert it to match the original format expected by downstream code
            
            // Note: We're returning the output_buf directly instead of padding it to match
            // the original complex FFT output size. The mel_spectrogram functions will need
            // to be adjusted accordingly.
            Some(Array1::from_vec(self.output_buf.clone()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        
        // Check that the output length is now fft_size/2 + 1 instead of fft_size
        assert_eq!(fft_frame.unwrap().len(), fft_size/2 + 1);
    }
}
