use crate::pcm::{apply_dither, apply_preemphasis};
use crate::{
    config::{LogType, MelConfig},
    mel::{ln_mel_spectrogram, log10_mel_spectrogram, mel},
    stft,
};
use ndarray::{concatenate, s, Array2, Axis};
use rtrb::{Consumer, PopError, Producer, PushError, RingBuffer as Rtrb};
use std::time::Instant;

/// A streaming mel-spectrogram generator over a fixed-capacity ring buffer
pub struct MelStreamer<C: MelConfig> {
    producer: Producer<f32>,
    consumer: Consumer<f32>,
    accumulated_samples: Vec<f32>,
    fft: stft::Spectrogram,
    filters: Array2<f64>,
    config: C,
    prev_sample: f32,
}

impl<C: MelConfig + Clone> MelStreamer<C> {
    /// Create a new MelStreamer with the given config and sample-capacity
    pub fn new(config: C, capacity: usize) -> Self {
        let hop = config.n_window_stride();
        let fft_size = config.n_fft();
        let sr = config.sample_rate() as f32;
        let n_mels = config.features();

        let (producer, consumer) = Rtrb::new(capacity);
        let accumulated_samples = Vec::with_capacity(hop);

        let filters = mel(
            config.sample_rate(),
            config.n_fft(),
            config.features(),
            None,
            None,
            false,
            true,
        );

        let fft = stft::Spectrogram::new(fft_size, hop);

        Self {
            producer,
            consumer,
            accumulated_samples,
            filters,
            fft,
            config,
            prev_sample: 0.0,
        }
    }

    /// Push an entire frame (slice) of new samples into the ring buffer
    pub fn add_frame(&mut self, samples: &[f32]) {
        for &s in samples {
            if let Err(PushError::Full(val)) = self.producer.push(s) {
                let _ = self.consumer.pop();
                let _ = self.producer.push(val);
            }
        }
    }

    /// Push a single new sample into the ring buffer
    pub fn add_sample(&mut self, sample: f32) {
        if let Err(PushError::Full(val)) = self.producer.push(sample) {
            let _ = self.consumer.pop();
            let _ = self.producer.push(val);
        }
    }

    /// If enough new samples have arrived to form one hop, produce one mel-spectrogram frame
    pub fn maybe_mel(&mut self) -> Option<Array2<f64>> {
        let hop = self.config.n_window_stride();

        while self.accumulated_samples.len() < hop {
            match self.consumer.pop() {
                Ok(s) => self.accumulated_samples.push(s),
                Err(PopError::Empty) => break,
            }
        }
        if self.accumulated_samples.len() < hop {
            return None;
        }

        let mut frame = std::mem::take(&mut self.accumulated_samples);

        if let Some(d) = self.config.dither() {
            apply_dither(&mut frame, d);
        }
        if let Some(p) = self.config.preemph() {
            apply_preemphasis(&mut frame, &mut self.prev_sample, p);
        }

        if let Some(fft) = self.fft.add(&frame) {
            let mel = match self.config.log_type() {
                LogType::Ln => ln_mel_spectrogram(&fft, &self.filters),
                LogType::Log10 => log10_mel_spectrogram(&fft, &self.filters),
            };
            Some(mel)
        } else {
            None
        }
    }

    /// Zero-pad and emit all remaining mel frames at end of input
    pub fn close(&mut self) -> Vec<Array2<f64>> {
        let mut out = Vec::new();
        let hop = self.config.n_window_stride();
        // inject zeros to flush partial window(s)
        for _ in 0..hop {
            self.add_sample(0.0);
        }
        // collect any frames produced by padding
        while let Some(frame) = self.maybe_mel() {
            out.push(frame);
        }
        out
    }
}
