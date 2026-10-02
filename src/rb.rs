use crate::{config::MelConfig, mel::MelSpectrogram, stft};
use ndarray::Array2;

#[cfg(feature = "rtrb")]
use rtrb::RingBuffer as RtrbBuffer;
#[cfg(feature = "rtrb")]
use rtrb::{Consumer, Producer, PushError};

#[cfg(not(feature = "rtrb"))]
use std::collections::VecDeque;

pub struct RingBuffer {
    accumulated_samples: Vec<f32>,
    #[cfg(not(feature = "rtrb"))]
    capacity: usize,

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
        assert!(capacity > 0, "capacity must be greater than zero");
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
            #[cfg(not(feature = "rtrb"))]
            capacity,
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
            for &s in samples {
                self.push_sample(s);
            }
        }
        #[cfg(not(feature = "rtrb"))]
        {
            if samples.len() >= self.capacity {
                self.buffer.clear();
                self.buffer
                    .extend(&samples[samples.len() - self.capacity..]);
                return;
            }

            let overflow = (self.buffer.len() + samples.len()).saturating_sub(self.capacity);
            if overflow > 0 {
                self.buffer.drain(..overflow);
            }
            self.buffer.extend(samples);
        }
    }

    pub fn add(&mut self, sample: f32) {
        self.push_sample(sample);
    }

    #[cfg(feature = "rtrb")]
    fn push_sample(&mut self, sample: f32) {
        if let Err(PushError::Full(sample)) = self.producer.push(sample) {
            let _ = self.consumer.pop();
            let result = self.producer.push(sample);
            debug_assert!(result.is_ok());
        }
    }

    #[cfg(not(feature = "rtrb"))]
    fn push_sample(&mut self, sample: f32) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(sample);
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
        let spectrum = self.fft.add_half(&self.accumulated_samples);
        self.accumulated_samples.clear();
        spectrum.map(|spectrum| self.mel.add_spectrum(spectrum, self.config.fft_size() / 2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::interleave_frames;
    use crate::test_support::{fuzz_cases, Rng};
    use ndarray::{Array2, Zip};
    use ndarray_npy::read_npy;
    use soundkit::{audio_bytes::deinterleave_vecs_f32, wav::WavStreamProcessor};
    use std::collections::VecDeque as ModelBuffer;
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

    #[test]
    fn oversized_write_retains_newest_samples() {
        let config = MelConfig::new(8, 4, 2, 16_000.0);
        let mut rb = RingBuffer::new(config, 4);
        rb.add_frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        #[cfg(feature = "rtrb")]
        let buffered = {
            let mut samples = Vec::new();
            while let Ok(sample) = rb.consumer.pop() {
                samples.push(sample);
            }
            samples
        };

        #[cfg(not(feature = "rtrb"))]
        let buffered = rb.buffer.iter().copied().collect::<Vec<_>>();

        assert_eq!(buffered, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn single_sample_write_discards_oldest_sample() {
        let config = MelConfig::new(8, 4, 2, 16_000.0);
        let mut rb = RingBuffer::new(config, 3);
        rb.add_frame(&[1.0, 2.0, 3.0]);
        rb.add(4.0);

        #[cfg(feature = "rtrb")]
        let buffered = {
            let mut samples = Vec::new();
            while let Ok(sample) = rb.consumer.pop() {
                samples.push(sample);
            }
            samples
        };

        #[cfg(not(feature = "rtrb"))]
        let buffered = rb.buffer.iter().copied().collect::<Vec<_>>();

        assert_eq!(buffered, vec![2.0, 3.0, 4.0]);
    }

    /// The ring buffer must give the frames of the public streaming API for
    /// random write sizes, overflows, and read patterns.
    #[test]
    fn fuzz_ringbuffer_matches_streaming_model() {
        let mut rng = Rng::new(10);
        for case in 0..fuzz_cases(80) {
            let fft_size = rng.range(8, 600);
            let hop_size = rng.range(1, fft_size);
            let n_mels = rng.range(1, 80);
            let capacity = rng.range(1, fft_size * 4);
            let config = MelConfig::new(fft_size, hop_size, n_mels, 16_000.0);
            let mut rb = RingBuffer::new(config, capacity);

            let mut buffer = ModelBuffer::new();
            let mut pending = Vec::new();
            let mut stft = stft::Spectrogram::new(fft_size, hop_size);
            let mut mel = MelSpectrogram::new(fft_size, 16_000.0, n_mels);

            for step in 0..rng.range(1, 60) {
                let write = rng.range(0, capacity * 2);
                let samples = rng.signal(write);
                rb.add_frame(&samples);
                for sample in samples {
                    if buffer.len() == capacity {
                        buffer.pop_front();
                    }
                    buffer.push_back(sample);
                }

                for read in 0..rng.range(0, 3) {
                    let got = rb.maybe_mel();
                    while pending.len() < hop_size {
                        match buffer.pop_front() {
                            Some(sample) => pending.push(sample),
                            None => break,
                        }
                    }
                    let want = if pending.len() < hop_size {
                        None
                    } else {
                        let frame = stft.add(&pending).map(|fft| mel.add(&fft));
                        pending.clear();
                        frame
                    };
                    assert_eq!(got, want, "case {case} step {step} read {read}");
                }
            }
        }
    }
}
