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
    use ndarray::{Array2, Zip};
    use ndarray_npy::read_npy;
    use soundkit::{audio_bytes::deinterleave_vecs_f32, wav::WavStreamProcessor};
    use std::fs::File;
    use std::io::Read;

    #[test]
    fn test_ringbuffer() {
        let fft_size = 512;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16_000.0;
        let config = MelConfig::new(fft_size, hop_size, n_mels, sampling_rate);
        let mut rb = RingBuffer::new(config, 1024);

        let mut file = File::open("./testdata/jfk_f32le.wav").unwrap();
        let mut processor = WavStreamProcessor::new();
        let mut buf = [0_u8; 128];
        let mut frames: Vec<Array2<f64>> = Vec::new();

        loop {
            let n = file.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            if let Ok(Some(audio)) = processor.add(&buf[..n]) {
                let samples = deinterleave_vecs_f32(audio.data(), 1);
                rb.add_frame(&samples[0]);
                if let Some(mel_frame) = rb.maybe_mel() {
                    frames.push(mel_frame);
                }
            }
        }

        // interleave and collect as f64
        let flat_f32: Vec<f32> = interleave_frames(&frames, false, 0);
        let flat: Vec<f64> = flat_f32.into_iter().map(f64::from).collect();

        let t = frames.len();
        let f = frames[0].dim().0;
        let got: Array2<f64> = Array2::from_shape_vec((f, t), flat).unwrap();

        // load golden as f32
        let want_f32: Array2<f32> = read_npy("./testdata/rust_jfk_golden.npy").unwrap();

        assert_eq!(got.shape(), want_f32.shape());

        Zip::from(&got).and(&want_f32).for_each(|&a_f64, &b_f32| {
            let a = a_f64 as f32;
            assert!((a - b_f32).abs() <= 1e-6);
        });
    }
}
