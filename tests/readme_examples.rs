use mel_spec::fbank::{Fbank, FbankConfig};
use mel_spec::mel::{interleave_frames, mel, MelSpectrogram};
use mel_spec::prelude::*;
use mel_spec::quant::{load_tga_8bit, save_tga_8bit};
use mel_spec::vad::{DetectionSettings, VadFrameTiming, VoiceActivityDetector};
use ndarray::Array2;
use num::Complex;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn readme_quick_start_cpu_mel_pipeline_runs() {
    let samples = vec![0.0_f32; 16_000];
    let mel_frames = Spectrogram::compute_mel_spectrogram_cpu(&samples, 400, 160, 80, 16_000.0);

    assert!(!mel_frames.is_empty());
    assert_eq!(mel_frames[0].len(), 80);
}

#[test]
fn readme_fbank_example_runs() {
    let samples = vec![0.0_f32; 16_000];
    let config = FbankConfig::default();
    let fbank = Fbank::new(config);

    let features = fbank.compute(&samples);

    assert_eq!(features.ncols(), 80);
    assert!(features.nrows() > 0);
}

#[test]
fn readme_mel_and_stft_examples_run() {
    let sampling_rate = 16_000.0;
    let fft_size = 400;
    let hop_size = 160;
    let n_mels = 80;

    let filters = mel(sampling_rate, fft_size, n_mels, None, None, false, true);
    assert_eq!(filters.shape(), &[80, 201]);

    let mut stft = Spectrogram::new(fft_size, hop_size);
    let mut mel_spec = MelSpectrogram::new(fft_size, sampling_rate, n_mels);

    let mut mel_frame = None;
    for _ in 0..3 {
        let samples = vec![0.0_f32; hop_size];
        if let Some(fft_frame) = stft.add(&samples) {
            mel_frame = Some(mel_spec.add(&fft_frame));
        }
    }

    assert_eq!(mel_frame.unwrap().shape(), &[80, 1]);
}

#[test]
fn readme_tga_example_runs() {
    let n_mels = 80;
    let frame_data = (0..n_mels).map(|idx| idx as f64 / n_mels as f64).collect();
    let frames = vec![Array2::from_shape_vec((n_mels, 1), frame_data).unwrap()];
    let interleaved = interleave_frames(&frames, false, 2);
    let path = temp_tga_path();

    save_tga_8bit(&interleaved, n_mels, path.to_str().unwrap()).unwrap();
    let loaded = load_tga_8bit(path.to_str().unwrap()).unwrap();
    fs::remove_file(path).unwrap();

    assert_eq!(loaded.len(), interleaved.len());
}

#[test]
fn readme_vad_timestamp_example_runs() {
    let settings = DetectionSettings::default();
    let timing = VadFrameTiming::new(400, 160, 16_000.0);
    let mut vad = VoiceActivityDetector::new_with_timing(&settings, timing);
    let mel_frame = Array2::from_shape_vec((80, 1), vec![0.0; 80]).unwrap();

    for _ in 0..settings.min_x {
        let _ = vad.add_activity(&mel_frame);
    }

    let activity = vad
        .add_activity(&mel_frame)
        .expect("VAD should emit once enough frames are buffered");
    assert!(activity.timestamps.is_some());
}

#[test]
fn readme_direct_fft_to_mel_example_runs() {
    let fft_size = 400;
    let sampling_rate = 16_000.0;
    let n_mels = 80;
    let mut mel_spec = MelSpectrogram::new(fft_size, sampling_rate, n_mels);

    let fft_input = ndarray::Array1::from(vec![Complex::new(1.0, 0.0); fft_size]);
    let mel_frame = mel_spec.add(&fft_input);

    assert_eq!(mel_frame.shape(), &[80, 1]);
}

fn temp_tga_path() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("mel-spec-readme-{nanos}.tga"))
}
