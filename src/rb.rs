use crate::{config::MelConfig, mel::MelSpectrogram, stft};
use ndarray::Array2;
use std::collections::VecDeque;

pub struct RingBuffer {
    accumulated_samples: Vec<f32>,
    buffer: VecDeque<f32>,
    fft: stft::Spectrogram,
    mel: MelSpectrogram,
    config: MelConfig,
}

impl RingBuffer {
    pub fn new(config: MelConfig, capacity: usize) -> Self {
        let hop_size = config.hop_size();
        let fft_size = config.fft_size();
        let sample_rate = config.sampling_rate();

        let buffer = VecDeque::with_capacity(capacity);
        let accumulated_samples = Vec::with_capacity(hop_size);
        let mel = MelSpectrogram::new(fft_size, sample_rate, config.n_mels());
        let fft = stft::Spectrogram::new(fft_size, hop_size);

        Self {
            config,
            buffer,
            accumulated_samples,
            mel,
            fft,
        }
    }

    pub fn add_frame(&mut self, samples: &[f32]) {
        let available_capacity = self.buffer.capacity() - self.buffer.len();
        if samples.len() > available_capacity {
            // If incoming samples exceed available capacity, remove the excess from the front
            self.buffer.drain(0..samples.len() - available_capacity);
        }
        self.buffer.extend(samples);
    }

    pub fn add(&mut self, sample: f32) {
        if self.buffer.len() == self.buffer.capacity() {
            self.buffer.pop_front();
        }
        self.buffer.push_back(sample);
    }

    pub fn maybe_mel(&mut self) -> Option<Array2<f64>> {
        let hop_size = self.config.hop_size();

        if self.accumulated_samples.len() >= hop_size {
            // Accumulated samples are sufficient to process
            let frame = self.accumulated_samples.split_off(hop_size);
            std::mem::swap(&mut self.accumulated_samples, &mut frame.clone());
        } else {
            // Accumulate more samples from the buffer
            let to_add = hop_size - self.accumulated_samples.len();
            let available = self.buffer.len().min(to_add); // Ensure we don't try to drain more than available
            self.accumulated_samples
                .extend(self.buffer.drain(..available));
        }

        if self.accumulated_samples.len() < hop_size {
            return None;
        }

        let fft_result = self.fft.add(&self.accumulated_samples);
        self.accumulated_samples.clear();

        fft_result.map(|fft| self.mel.add(&fft))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::interleave_frames;
    use ndarray::{Array1, Axis};
    use ndarray_npy::write_npy;
    use soundkit::{audio_bytes::deinterleave_vecs_f32, wav::WavStreamProcessor};
    use std::fs::File;
    use std::io::Read;

    #[test]
    fn test_ringbuffer() {
        let fft_size = 512;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16000.0;

        let config = MelConfig::new(fft_size, hop_size, n_mels, sampling_rate);

        let mut rb = RingBuffer::new(config, 1024);

        // 16000 Hz, mono, flt
        let file_path = "./testdata/jfk_f32le.wav";
        let mut file = File::open(&file_path).unwrap();

        let mut processor = WavStreamProcessor::new();
        let mut buffer = [0u8; 128];

        let mut frames: Vec<Array2<f64>> = Vec::new();
        loop {
            let bytes_read = file.read(&mut buffer).unwrap();
            if bytes_read == 0 {
                break;
            }

            let chunk = &buffer[..bytes_read];
            match processor.add(chunk) {
                Ok(Some(audio_data)) => {
                    // the test data is f32 mono
                    let samples = deinterleave_vecs_f32(audio_data.data(), 1);
                    rb.add_frame(&samples[0]);
                    if let Some(mel_frame) = rb.maybe_mel() {
                        frames.push(mel_frame);
                    }
                }
                Ok(None) => continue,
                Err(err) => panic!("Error: {}", err),
            }
        }

        let flattened_frames = interleave_frames(&frames, false, 0);

        let num_time_steps = frames.len();
        let num_frequency_bands = frames[0].dim().0; // Assuming all frames have the same number of rows

        // Reshape the flattened frames into a 2D array
        let stacked_frames =
            Array2::from_shape_vec((num_frequency_bands, num_time_steps), flattened_frames)
                .expect("Error reshaping flattened frames");

        write_npy("./testdata/rust_jfk.npy", &stacked_frames);
    }
}
