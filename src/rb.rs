use crate::{config::MelConfig, mel::MelSpectrogram, stft};
use ndarray::Array2;

#[cfg(feature = "rtrb")]
use rtrb::RingBuffer as RtrbBuffer;
#[cfg(feature = "rtrb")]
use rtrb::{Consumer, Producer};

#[cfg(not(feature = "rtrb"))]
use std::collections::VecDeque;

pub struct RingBuffer {
    accumulated_samples: Vec<f32>,

    #[cfg(feature = "rtrb")]
    // rtrb gives us a single-producer/single-consumer buffer
    producer: Producer<f32>,
    #[cfg(feature = "rtrb")]
    consumer: Consumer<f32>,

    #[cfg(not(feature = "rtrb"))]
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

        #[cfg(feature = "rtrb")]
        let (producer, consumer) = RtrbBuffer::<f32>::new(capacity);

        #[cfg(not(feature = "rtrb"))]
        let buffer = VecDeque::with_capacity(capacity);

        Self {
            config: config.clone(),
            accumulated_samples: Vec::with_capacity(hop_size),
            #[cfg(feature = "rtrb")]
            producer,
            #[cfg(feature = "rtrb")]
            consumer,
            #[cfg(not(feature = "rtrb"))]
            buffer,
            fft: stft::Spectrogram::new(fft_size, hop_size),
            mel: MelSpectrogram::new(fft_size, sample_rate, config.n_mels()),
        }
    }

    pub fn add_frame(&mut self, samples: &[f32]) {
        #[cfg(feature = "rtrb")]
        {
            // rtrb::Producer::push will overwrite old data if full
            for &s in samples {
                let _ = self.producer.push(s);
            }
        }
        #[cfg(not(feature = "rtrb"))]
        {
            let available = self.buffer.capacity() - self.buffer.len();
            if samples.len() > available {
                self.buffer.drain(0..(samples.len() - available));
            }
            self.buffer.extend(samples);
        }
    }

    pub fn add(&mut self, sample: f32) {
        #[cfg(feature = "rtrb")]
        {
            let _ = self.producer.push(sample);
        }
        #[cfg(not(feature = "rtrb"))]
        {
            if self.buffer.len() == self.buffer.capacity() {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }
    }

    pub fn maybe_mel(&mut self) -> Option<Array2<f64>> {
        let hop_size = self.config.hop_size();

        // first, accumulate into `accumulated_samples`
        #[cfg(feature = "rtrb")]
        {
            while self.accumulated_samples.len() < hop_size {
                if let Ok(s) = self.consumer.pop() {
                    self.accumulated_samples.push(s);
                } else {
                    break;
                }
            }
        }
        #[cfg(not(feature = "rtrb"))]
        {
            let to_add = hop_size - self.accumulated_samples.len();
            let available = self.buffer.len().min(to_add);
            self.accumulated_samples
                .extend(self.buffer.drain(..available));
        }

        if self.accumulated_samples.len() < hop_size {
            return None;
        }

        // we have enough to do one frame
        let mut frame = Vec::new();
        std::mem::swap(&mut frame, &mut self.accumulated_samples);

        let fft_res = self.fft.add(&frame);
        match fft_res {
            Some(fft) => Some(self.mel.add(&fft)),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::interleave_frames;
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

        let _ = write_npy("./testdata/rust_jfk.npy", &stacked_frames);
    }
}
