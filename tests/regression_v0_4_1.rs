//! Compares current outputs with fixtures that mel_spec 0.4.1 wrote for the JFK
//! sample. The fixtures are in `testdata/regression`.

use mel_spec::fbank::{Fbank, FbankConfig};
use mel_spec::mel::{BatchLogMelConfig, BatchLogMelSpectrogram, MelSpectrogram};
use mel_spec::stft::Spectrogram;
use mel_spec::vad::{DetectionSettings, VoiceActivityDetector};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// The first three seconds of the JFK sample, as in the fixtures.
const FIXTURE_SAMPLES: usize = 48_000;

fn load_jfk() -> Vec<f32> {
    let bytes = std::fs::read("./testdata/jfk_f32le.wav").unwrap();
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if &bytes[pos..pos + 4] == b"data" {
            return bytes[pos + 8..pos + 8 + size]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
        }
        pos += 8 + size + (size & 1);
    }
    panic!("JFK sample has no data chunk");
}

fn fixture(name: &str) -> Array2<f32> {
    read_npy(format!("./testdata/regression/{name}")).unwrap()
}

/// Maximum and mean absolute difference.
fn differences(got: &Array2<f32>, want: &Array2<f32>) -> (f32, f32) {
    assert_eq!(got.dim(), want.dim());
    let diffs = got
        .iter()
        .zip(want.iter())
        .map(|(got, want)| (got - want).abs())
        .collect::<Vec<_>>();
    let max = diffs.iter().copied().fold(0.0_f32, f32::max);
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    (max, mean)
}

// The fixtures come from one platform. Math libraries and FFT kernels on other
// platforms can change the last bits, so the float checks have tolerances.
// On the fixture platform, Fbank and the Whisper batch path match exactly.

#[test]
fn fbank_matches_v0_4_1() {
    let samples = load_jfk();
    let got = Fbank::new(FbankConfig::default()).compute(&samples[..FIXTURE_SAMPLES]);
    let (max, _) = differences(&got, &fixture("v0_4_1_fbank_jfk_3s.npy"));
    assert!(max <= 1e-4, "maximum difference {max}");
}

#[test]
fn whisper_batch_mel_matches_v0_4_1() {
    let samples = load_jfk();
    let frames = Spectrogram::compute_mel_spectrogram_cpu(
        &samples[..FIXTURE_SAMPLES],
        400,
        160,
        80,
        16_000.0,
    );
    let got = Array2::from_shape_vec((frames.len(), 80), frames.concat()).unwrap();
    let (max, _) = differences(&got, &fixture("v0_4_1_whisper_batch_jfk_3s.npy"));
    assert!(max <= 1e-5, "maximum difference {max}");
}

/// The real-input FFT changes the FFT rounding, so this path has a tolerance.
/// On this fixture the measured maximum difference is 3.4e-4 and the mean
/// difference is 1.2e-6.
#[test]
fn parakeet_batch_log_mel_is_within_fft_rounding_of_v0_4_1() {
    let samples = load_jfk();
    let frontend = BatchLogMelSpectrogram::new(BatchLogMelConfig {
        n_mels: 128,
        preemphasis: 0.97,
        log_zero_guard: 2.0_f32.powi(-24),
        normalize_per_feature: true,
        f_max: Some(8_000.0),
        ..BatchLogMelConfig::default()
    })
    .unwrap();
    let got = frontend.compute(&samples[..FIXTURE_SAMPLES]).unwrap();
    let (max, mean) = differences(&got, &fixture("v0_4_1_parakeet_jfk_3s.npy"));
    eprintln!("parakeet difference from 0.4.1: max {max:e}, mean {mean:e}");
    assert!(max <= 5e-3, "maximum difference {max}");
    assert!(mean <= 1e-5, "mean difference {mean}");
}

#[test]
fn streaming_vad_decisions_match_v0_4_1() {
    let samples = load_jfk();
    let mut stft = Spectrogram::new(400, 160);
    let mut mel = MelSpectrogram::new(400, 16_000.0, 80);
    let mut vad = VoiceActivityDetector::new(&DetectionSettings::default());
    let mut rows = Vec::new();
    for chunk in samples.chunks(160) {
        if let Some(fft) = stft.add(chunk) {
            if let Some(activity) = vad.add_activity(&mel.add(&fft)) {
                rows.extend([
                    activity.frame_index as f64,
                    f64::from(u8::from(activity.active)),
                    activity.leading_active_columns as f64,
                    activity.active_columns as f64,
                    activity.window_columns as f64,
                    activity.confidence,
                ]);
            }
        }
    }
    let got = Array2::from_shape_vec((rows.len() / 6, 6), rows).unwrap();
    let want: Array2<f64> = read_npy("./testdata/regression/v0_4_1_vad_jfk.npy").unwrap();
    assert_eq!(got, want);
}
