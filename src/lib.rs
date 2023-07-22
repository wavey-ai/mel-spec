use ndarray::{concatenate, s, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1};
use num::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

pub struct AudioProcessor {
    hop_buf: Vec<f64>,
    complex_buf: Vec<Complex<f64>>,
    scratch_buf: Vec<Complex<f64>>,
    fft: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
    hop_size: usize,
    fft_size: usize,
}

// stream-based stft processor.
// Nearly identical to whisper.cpp, pytorch, etc, but the caller might be mindful of the
// first and final frames:
//   a) pass in exact FTT-size sample for initial window to avoid automatic zero-padding
//   b) be aware the final frame will be zero-padded if it is < hop size.
//     - neither is necessary unless you are running additional analysis.
impl AudioProcessor {
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - f64::cos((2.0 * PI * i as f64) / fft_size as f64)))
            .collect();

        Self {
            hop_buf: vec![0.0; fft_size],
            complex_buf: vec![Complex::new(0.0, 0.0); fft_size],
            scratch_buf: vec![Complex::new(0.0, 0.0); fft_size],
            fft,
            window,
            hop_size,
            fft_size,
        }
    }

    // add new samples, get an FFT out.
    // ~580 microseconds (ftt_size=400) on M2 Air.
    pub fn add(&mut self, data: &Vec<f32>) -> Array1<Complex<f64>> {
        let fft_size = self.fft_size;
        let hop_size = self.hop_size;
        let mut pcm_data: Vec<f64> = data.iter().map(|x| *x as f64).collect();
        let pcm_size = pcm_data.len();

        if pcm_size < hop_size {
            pcm_data.extend_from_slice(&vec![0.0; hop_size - pcm_size]);
        }

        if pcm_size == fft_size {
            self.hop_buf.copy_from_slice(&pcm_data);
        } else if pcm_size == hop_size {
            self.hop_buf.copy_within(hop_size.., 0);
            self.hop_buf[(fft_size - hop_size)..].copy_from_slice(&pcm_data);
        }

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

        Array1::from_vec(self.complex_buf.clone())
    }
}

// Rust port of whisper.cpp and whisper.py log_mel_spectrogram
// NB: a) normalisation is a separate step, refer to norm_mel.
//     b) the Array2 output must be correctly interleaved before processing with
//      whisper.cpp - refer to interleave_frame.
pub fn log_mel_spectrogram(stft: &Array1<Complex<f64>>, mel_filters: &Array2<f64>) -> Array2<f64> {
    let mut magnitudes_padded = stft
        .iter()
        .map(|v| v.norm_sqr())
        .take(stft.len() / 2)
        .collect::<Vec<_>>();

    magnitudes_padded.push(0.0);

    let magnitudes_reshaped =
        Array2::from_shape_vec((1, magnitudes_padded.len()), magnitudes_padded).unwrap();

    let epsilon = 1e-10;
    let mel_spec = mel_filters
        .dot(&magnitudes_reshaped.t())
        .mapv(|sum| (sum.max(epsilon)).log10());

    mel_spec
}

// Normalisation based on max value in the sample window.
//
// Two variants of this function are provided, only one should be used in a pipeline:
//   norm_mel: to be called on individual Array2 results from log_mel_spectrogram
//   norm_mel_vec: to be called on the product of interleave_frames
//
// One of these must be applied to the individual or combined output of log_mel_spectrogram
// before processing with whisper (note that it also converts from f64 precision to f32).
//
// It's adequate to normalise ftt window-size sample lengths individually but larger sample
// sizes may sometimes give better results and these functions allow flexibility in the
// sample size that's normalised over.
pub fn norm_mel(mel_spec: &Array2<f64>) -> Array2<f64> {
    let mmax = mel_spec.fold(f64::NEG_INFINITY, |acc, x| acc.max(*x));
    let mmax = mmax - 8.0;
    let clamped: Array2<f64> = mel_spec.mapv(|x| (x.max(mmax) + 4.0) / 4.0).mapv(|x| x);

    clamped
}

// vector-variant of norm_mel.
pub fn norm_mel_vec(mel_spec: &[f32]) -> Vec<f32> {
    let mmax = mel_spec
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let mmax = mmax - 8.0;
    let clamped: Vec<f32> = mel_spec
        .iter()
        .map(|&x| ((x.max(mmax) + 4.0) / 4.0) as f32)
        .collect();

    clamped
}

// Interleave in major row order: ideal of waterfall - each row is a scanline.
// Interleave in major column order: Data will be interleaved such that each row
//   represents a different frequency band, and each column represents a time step,
//   matching the format of the whisper.cpp C mel.data array.
pub fn interleave_frames(frames: &Vec<Array2<f64>>, major_row_order: bool) -> Vec<f32> {
    let num_frames = frames.len();
    let num_filters = frames[0].shape()[0];

    let mut interleaved_data = Vec::with_capacity(num_frames * num_filters);

    if major_row_order {
        // Interleave in major row order
        for frame_idx in 0..num_frames {
            for filter_idx in 0..num_filters {
                let frame_view = ArrayView2::from(&frames[frame_idx]);
                interleaved_data.push(*frame_view.get((filter_idx, 0)).unwrap() as f32);
            }
        }
    } else {
        // Interleave in major column order - whisper.cpp
        for filter_idx in 0..num_filters {
            for frame_idx in 0..num_frames {
                let frame_view = ArrayView2::from(&frames[frame_idx]);
                interleaved_data.push(*frame_view.get((filter_idx, 0)).unwrap() as f32);
            }
        }
    }

    interleaved_data
}

// mel filterbanks, within 1.0e-7 of librosa and identical to whisper GGML model-embedded filters.
pub fn mel(sr: f64, n_fft: usize, n_mels: usize, hkt: bool, norm: bool) -> Array2<f64> {
    let fftfreqs = fft_frequencies(sr, n_fft);
    let f_min: f64 = 0.0; // Minimum frequency
    let f_max: f64 = sr / 2.0; // Maximum frequency
    let mel_f = mel_frequencies(n_mels + 2, f_min, f_max, hkt);

    // calculate the triangular mel filter bank weights for mel-frequency cepstral coefficient (MFCC) computation
    let fdiff = &mel_f.slice(s![1..n_mels + 2]) - &mel_f.slice(s![..n_mels + 1]);
    let ramps = &mel_f.slice(s![..n_mels + 2]).insert_axis(Axis(1)) - &fftfreqs;

    let mut weights = Array2::zeros((n_mels, n_fft / 2 + 1));

    for i in 0..n_mels {
        let lower = -&ramps.row(i) / fdiff[i];
        let upper = &ramps.row(i + 2) / fdiff[i + 1];

        weights
            .row_mut(i)
            .assign(&lower.mapv(|x| x.max(0.0).min(1.0)));

        weights
            .row_mut(i)
            .zip_mut_with(&upper.mapv(|x| x.max(0.0).min(1.0)), |a, &b| {
                *a = (*a).min(b);
            });
    }

    if norm {
        // Slaney-style mel is scaled to be approx constant energy per channel
        let enorm = 2.0 / (&mel_f.slice(s![2..n_mels + 2]) - &mel_f.slice(s![..n_mels]));
        weights *= &enorm.insert_axis(Axis(1));
    }

    weights
}

// it's unclear if this is required - whisper.cpp works fine without padding.
pub fn pad_or_trim(array: &Array2<f32>, length: usize) -> Array2<f32> {
    let array_shape = array.shape();
    let original_length = array_shape[1];

    if original_length > length {
        return array.slice(s![.., ..length]).to_owned();
    } else if original_length < length {
        let pad_width = length - original_length;
        let pad_array = Array2::<f32>::zeros((array_shape[0], pad_width));
        return concatenate![Axis(1), array.view(), pad_array.view()].to_owned();
    }

    array.to_owned()
}

fn hz_to_mel(frequency: f64, htk: bool) -> f64 {
    if htk {
        return 2595.0 * (1.0 + (frequency / 700.0).log10());
    }

    let f_min: f64 = 0.0;
    let f_sp: f64 = 200.0 / 3.0;
    let min_log_hz: f64 = 1000.0;
    let min_log_mel: f64 = (min_log_hz - f_min) / f_sp;
    let logstep: f64 = (6.4f64).ln() / 27.0;

    if frequency >= min_log_hz {
        min_log_mel + ((frequency / min_log_hz).ln() / logstep)
    } else {
        (frequency - f_min) / f_sp
    }
}

fn mel_to_hz(mel: f64, htk: bool) -> f64 {
    if htk {
        return 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0);
    }

    let f_min: f64 = 0.0;
    let f_sp: f64 = 200.0 / 3.0;
    let min_log_hz: f64 = 1000.0;
    let min_log_mel: f64 = (min_log_hz - f_min) / f_sp;
    let logstep: f64 = (6.4f64).ln() / 27.0;

    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

fn mels_to_hz(mels: ArrayBase<impl Data<Elem = f64>, Ix1>, htk: bool) -> Array1<f64> {
    mels.mapv(|mel| mel_to_hz(mel, htk))
}

fn mel_frequencies(n_mels: usize, fmin: f64, fmax: f64, htk: bool) -> Array1<f64> {
    let min_mel = hz_to_mel(fmin, htk);
    let max_mel = hz_to_mel(fmax, htk);

    let mels = Array1::linspace(min_mel, max_mel, n_mels);
    mels_to_hz(mels, htk)
}

fn fft_frequencies(sr: f64, n_fft: usize) -> Array1<f64> {
    let step = sr / n_fft as f64;
    let freqs: Array1<f64> = Array1::from_shape_fn(n_fft / 2 + 1, |i| step * i as f64);
    freqs
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::NpzReader;
    use std::fs::File;
    use std::io::Read;
    //    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

    struct AudioFileData {
        bits_per_sample: u8,
        channel_count: u8,
        data: Vec<u8>,
        sampling_rate: u32,
    }

    fn parse_wav<R: Read>(mut reader: R) -> Result<AudioFileData, String> {
        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .map_err(|err| err.to_string())?;

        if &buffer[..4] != b"RIFF" || &buffer[8..12] != b"WAVE" {
            return Err("Not a WAV file".to_string());
        }

        let mut position = 12; // After "WAVE"

        while &buffer[position..position + 4] != b"fmt " {
            let chunk_size =
                u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
            position += chunk_size + 8; // Advance to next chunk
        }

        let fmt_chunk = &buffer[position..position + 24];
        let sampling_rate = u32::from_le_bytes(fmt_chunk[12..16].try_into().unwrap()) as u32;
        let bits_per_sample = u16::from_le_bytes(fmt_chunk[22..24].try_into().unwrap()) as u8;
        let channel_count = u16::from_le_bytes(fmt_chunk[10..12].try_into().unwrap()) as u8;

        // Move position to after "fmt " chunk
        let chunk_size =
            u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
        position += chunk_size + 8;

        while &buffer[position..position + 4] != b"data" {
            let chunk_size =
                u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
            position += chunk_size + 8; // Advance to next chunk
        }

        let data_chunk = buffer[position + 8..].to_vec(); // Skip "data" and size

        let result = AudioFileData {
            bits_per_sample,
            channel_count,
            sampling_rate,
            data: data_chunk,
        };

        Ok(result)
    }

    pub fn deinterleave_vecs_f32(input: &[u8], channel_count: usize) -> Vec<Vec<f32>> {
        let sample_size = input.len() / (channel_count * 4);
        let mut result = vec![vec![0.0; sample_size]; channel_count];

        for i in 0..sample_size {
            for channel in 0..channel_count {
                let start = (i * channel_count + channel) * 4;
                let value = f32::from_le_bytes(input[start..start + 4].try_into().unwrap());
                result[channel][i] = value;
            }
        }

        result
    }

    macro_rules! assert_nearby {
        ($left:expr, $right:expr, $epsilon:expr) => {{
            let (left_val, right_val) = (&$left, &$right);
            assert_eq!(
                left_val.len(),
                right_val.len(),
                "Arrays have different lengths"
            );

            for (l, r) in left_val.iter().zip(right_val.iter()) {
                let diff = (*l - *r).abs();
                assert!(
                    diff <= $epsilon,
                    "Assertion failed: left={}, right={}, epsilon={}",
                    l,
                    r,
                    $epsilon
                );
            }
        }};
    }

    #[test]
    fn test_hz_to_mel() {
        let got = vec![hz_to_mel(60.0, false); 1];
        let want = vec![0.9; 1];
        assert_nearby!(got, want, 0.001);
    }

    #[test]
    fn test_mel_to_hz() {
        assert_eq!(mel_to_hz(3.0, false), 200.0);
    }

    #[test]
    fn test_mels_to_hz() {
        let mels = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let want = Array1::from(vec![66.667, 133.333, 200., 266.667, 333.333]);
        let got = mels_to_hz(mels, false);
        assert_nearby!(got, want, 0.001);
    }

    #[test]
    fn test_mel_frequencies() {
        let n_mels = 40;
        let fmin = 0.0;
        let fmax = 11025.0; // librosa.mel_frequencies default max val

        // taken from librosa.mel_frequencies(n_mels=40) in-line example
        let want = Array1::from(vec![
            0., 85.317, 170.635, 255.952, 341.269, 426.586, 511.904, 597.221, 682.538, 767.855,
            853.173, 938.49, 1024.856, 1119.114, 1222.042, 1334.436, 1457.167, 1591.187, 1737.532,
            1897.337, 2071.84, 2262.393, 2470.47, 2697.686, 2945.799, 3216.731, 3512.582, 3835.643,
            4188.417, 4573.636, 4994.285, 5453.621, 5955.205, 6502.92, 7101.009, 7754.107,
            8467.272, 9246.028, 10096.408, 11025.,
        ]);
        let got = mel_frequencies(n_mels, fmin, fmax, false);
        assert_nearby!(got, want, 0.005);
    }

    #[test]
    fn test_fft_frequences() {
        let sr = 22050.0;
        let n_fft = 16;

        // librosa.fft_frequencies(sr=22050, n_fft=16)
        let want = Array1::from(vec![
            0., 1378.125, 2756.25, 4134.375, 5512.5, 6890.625, 8268.75, 9646.875, 11025.,
        ]);
        let got = fft_frequencies(sr, n_fft);
        assert_nearby!(got, want, 0.001);
    }

    #[test]
    fn test_mel() {
        // whisper mel filterbank
        let file_path = "test/mel_filters.npz";
        let f = File::open(file_path).unwrap();
        let mut npz = NpzReader::new(f).unwrap();
        let filters: Array2<f32> = npz.by_index(0).unwrap();
        let want: Array2<f64> = filters.mapv(|x| f64::from(x));
        let got = mel(16000.0, 400, 80, false, true);
        assert_eq!(got.shape(), vec![80, 201]);
        for i in 0..80 {
            assert_nearby!(got.row(i), want.row(i), 1.0e-7);
        }
    }

    #[test]
    fn test_spec() {
        // load the whisper jfk sample
        let file_path = "test/jfk_f32le.wav";
        let file = File::open(&file_path).unwrap();
        let data = parse_wav(file).unwrap();
        assert_eq!(data.sampling_rate, 16000);
        assert_eq!(data.channel_count, 1);
        assert_eq!(data.bits_per_sample, 32);

        let samples = deinterleave_vecs_f32(&data.data, 1);

        // setup stft
        let fft_size = 400;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16000.0;
        let mut fft = AudioProcessor::new(fft_size, hop_size);

        // setup mel filterbank
        let filters = mel(sampling_rate, fft_size, n_mels, false, true);
        let mut mels: Vec<Array2<f64>> = Vec::new();

        // process samples - note the caller is responsioble for sending samples in chunks of
        // whisper's fft hop size (160 samples).
        for chunk in samples[0].chunks(hop_size) {
            let complex = fft.add(&chunk.to_vec());
            let mel = log_mel_spectrogram(&complex, &filters);
            // its sufficent to normalise this example in very small chunks of hop size but this
            // might not always be the most appropriate sample size.
            let norm = norm_mel(&mel);
            mels.push(norm);
        }

        // alternatively, you could normalise the interleaved frames here.
        // let mel_spectrogram = interleave_frames(&mels);

        // add whisper-rs to continue this example

        /*
                // let's send the mel spectrogram straight to the whisper model, by-passing any audio
                // processing in C
                let ctx = WhisperContext::new("../../whisper.cpp/models/ggml-medium.en.bin")
                    .expect("failed to load model");
                let mut state = ctx.create_state().expect("failed to create key");

                // set the spectrogram directly to the whisper state
                state.set_mel(&mel_spectrogram).unwrap();
                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

                // empty audio - whisper.cpp won't overwrite the mel state unless there are audio samples.
                let empty = vec![0.0; 0];
                state.full(params, &empty[..]).unwrap();

                let num_segments = state
                    .full_n_segments()
                    .expect("failed to get number of segments");
                for i in 0..num_segments {
                    let got = state
                        .full_get_segment_text(i)
                        .expect("failed to get segment");

                    assert_eq!(got, "[_BEG_] And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.[_TT_550]");
                }
        */
    }
}
