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
        let (producer, consumer) = Rtrb::new(capacity);
        let accumulated_samples = Vec::with_capacity(config.n_window_stride());

        let filters = mel(
            config.sample_rate(),
            config.n_fft(),
            config.features(),
            config.lowfreq(),
            config.highfreq(),
            false,
            config.mel_norm(),
        );

        let fft = stft::Spectrogram::new(config.n_fft(), config.n_window_stride());

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NemoConfig;
    use crate::mel::interleave_frames;
    use crate::quant::save_tga_8bit;
    use ndarray::{concatenate, s, Array2, Axis};
    use ndarray_npy::{read_npy, write_npy};
    use soundkit::{
        audio_bytes::{deinterleave_vecs_f32, deinterleave_vecs_i16},
        wav::WavStreamProcessor,
    };
    use std::{
        fs,
        fs::File,
        io::Read,
        path::{Path, PathBuf},
        time::Instant,
    };

    #[test]
    fn test_load_tensor_to_tga() {
        let start = Instant::now();

        let tensor_path = "./testdata/exported_audio.npy";
        let output_tga_path = "./testdata/exported_spectrogram.tga";

        let mut features: Array2<f32> =
            read_npy(Path::new(tensor_path)).expect("failed to read tensor .npy");
        let (mut n_mels, mut time_steps) = features.dim();
        if n_mels != 80 {
            features = features.reversed_axes().to_owned();
            let dims = features.dim();
            n_mels = dims.0;
            time_steps = dims.1;
        }
        assert_eq!(n_mels, 80);

        let features_f64 = features.mapv(|x| x as f64);
        let mut frames: Vec<Array2<f64>> = Vec::with_capacity(time_steps);
        for t in 0..time_steps {
            let frame = features_f64
                .slice(s![.., t])
                .to_owned()
                .into_shape((n_mels, 1))
                .unwrap();
            frames.push(frame);
        }

        let flattened = interleave_frames(&frames, false, 0);
        save_tga_8bit(&flattened, n_mels, output_tga_path).unwrap();

        let duration_ms = start.elapsed().as_millis();
        println!("test_load_tensor_to_tga took {} ms", duration_ms);
    }

    #[test]
    fn test_harvard_wavs_to_tga() {
        let test_start = Instant::now();

        let config = NemoConfig::default();
        let dir: &Path = Path::new("/Users/jamieb/wavey.ai/harvard-lines/output2/");
        let out: &Path = Path::new("./harvard");
        if !out.exists() {
            fs::create_dir_all(out).unwrap();
        }

        let files: Vec<PathBuf> = find_wav_files(dir);
        assert!(!files.is_empty(), "No WAV files found");

        for path in files {
            let file_start = Instant::now();

            let mut f = File::open(&path).unwrap();
            let mut proc = WavStreamProcessor::new();
            let mut stream = MelStreamer::new(config.clone(), 1024);
            let mut buf = [0u8; 1024];
            let mut frames: Vec<Array2<f64>> = Vec::new();

            while let Ok(n) = f.read(&mut buf) {
                if n == 0 {
                    break;
                }
                if let Ok(Some(data)) = proc.add(&buf[..n]) {
                    let i16s = deinterleave_vecs_i16(data.data(), 1);
                    let floats: Vec<f32> = i16s[0].iter().map(|&s| s as f32 / 32768.0).collect();
                    stream.add_frame(&floats);
                    while let Some(mel_frame) = stream.maybe_mel() {
                        frames.push(mel_frame);
                    }
                }
            }
            frames.extend(stream.close());

            let flattened = interleave_frames(&frames, false, 0);
            let filename = path.file_stem().unwrap().to_string_lossy();
            let output_path = out.join(format!("{}.tga", filename));
            save_tga_8bit(&flattened, config.features(), output_path.to_str().unwrap()).unwrap();

            let file_duration_ms = file_start.elapsed().as_millis();
            println!(
                "Processed '{}' in {} ms",
                path.file_name().unwrap().to_string_lossy(),
                file_duration_ms
            );
        }

        let total_duration_ms = test_start.elapsed().as_millis();
        println!(
            "test_harvard_wavs_to_tga all sentences took {} ms",
            total_duration_ms
        );
    }

    #[test]
    fn test_melstreamer_end_to_end() {
        let start = Instant::now();

        let mut file = File::open("./testdata/jfk_f32le.wav").unwrap();
        let mut proc = WavStreamProcessor::new();
        let config = NemoConfig::default();
        let mut stream = MelStreamer::new(config.clone(), 1024);
        let mut buf = [0u8; 128];
        let mut frames: Vec<Array2<f64>> = Vec::new();

        while let Ok(n) = file.read(&mut buf) {
            if n == 0 {
                break;
            }
            if let Ok(Some(data)) = proc.add(&buf[..n]) {
                let samps = deinterleave_vecs_f32(data.data(), 1);
                stream.add_frame(&samps[0]);
                while let Some(mel_frame) = stream.maybe_mel() {
                    frames.push(mel_frame);
                }
            }
        }
        frames.extend(stream.close());

        let flattened = interleave_frames(&frames, false, 0);
        let bands = frames[0].dim().0;
        let steps = frames.len();
        let stacked = Array2::from_shape_vec((bands, steps), flattened).unwrap();
        write_npy("./testdata/rust_jfk.npy", &stacked).unwrap();

        let duration_ms = start.elapsed().as_millis();
        println!("test_melstreamer_end_to_end took {} ms", duration_ms);
    }

    fn find_wav_files(dir: &Path) -> Vec<PathBuf> {
        let mut wav = Vec::new();
        if let Ok(entries) = fs::read_dir(dir) {
            for e in entries.filter_map(Result::ok) {
                let p = e.path();
                if p.is_dir() {
                    wav.extend(find_wav_files(&p));
                } else if p.extension().map(|e| e == "wav").unwrap_or(false) {
                    wav.push(p);
                }
            }
        }
        wav
    }
}
