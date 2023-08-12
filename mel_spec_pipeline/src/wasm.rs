use js_sys::{Array, Object, Reflect, Uint8Array, Uint8ClampedArray};
use mel_spec::mel::{interleave_frames, log_mel_spectrogram, mel, norm_mel};
use mel_spec::quant::quantize;
use mel_spec::stft::Spectrogram;
use mel_spec::vad::{duration_ms_for_n_frames, DetectionSettings, VoiceActivityDetector};
use ndarray::Array2;
use wasm_bindgen::prelude::*;
use web_sys::Worker;

#[wasm_bindgen]
pub struct SpeechToMel {
    mel: Array2<f64>,
    mel_vad: Array2<f64>,
    fft: Spectrogram,
    vad: VoiceActivityDetector,
    hop_size: usize,
    sampling_rate: f64,
    accumulated_samples: Vec<f32>,
    idx: usize,
}

#[wasm_bindgen]
impl SpeechToMel {
    #[wasm_bindgen]
    pub fn new(fft_size: usize, hop_size: usize, sampling_rate: f64, n_mels: usize) -> Self {
        let filters = mel(sampling_rate, fft_size, n_mels, false, true);
        let filters2 = mel(sampling_rate, fft_size, n_mels / 4, false, true);
        let stft = Spectrogram::new(fft_size, hop_size);
        let settings = DetectionSettings {
            min_energy: 1.0,
            min_y: 3,
            min_x: 3,
            min_mel: 0,
        };

        let vad = VoiceActivityDetector::new(&settings);
        Self {
            accumulated_samples: Vec::new(),
            mel: filters,
            mel_vad: filters2,
            fft: stft,
            vad,
            sampling_rate,
            hop_size,
            idx: 0,
        }
    }

    #[wasm_bindgen]
    pub fn get(&mut self) -> JsValue {
        let empty = vec![0.0; 0];
        self.add(empty)
    }

    #[wasm_bindgen]
    pub fn add(&mut self, data: Vec<f32>) -> JsValue {
        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(false)).unwrap();
        self.accumulated_samples.extend_from_slice(&data);
        if self.accumulated_samples.len() >= self.hop_size {
            let (samples, rest) = self.accumulated_samples.split_at(self.hop_size);

            Reflect::set(
                &result,
                &JsValue::from_str("len"),
                &JsValue::from(samples.len()),
            )
            .unwrap();

            if let Some(fft) = self.fft.add(&samples.to_vec()) {
                let frame = norm_mel(&log_mel_spectrogram(&fft, &self.mel));
                let frame2 = norm_mel(&log_mel_spectrogram(&fft, &self.mel_vad));
                let (quant_frame, _) = quantize(&interleave_frames(&[frame.clone()], false, 0));
                let frame_array = Uint8Array::from(&quant_frame[..]);
                let frame_clamped_array = Uint8ClampedArray::new(&frame_array.buffer());
                Reflect::set(&result, &JsValue::from_str("frame"), &frame_clamped_array).unwrap();
                Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(true)).unwrap();
                if let Some(gap) = self.vad.add(&frame2) {
                    let ms = duration_ms_for_n_frames(self.hop_size, self.sampling_rate, self.idx);
                    Reflect::set(&result, &JsValue::from_str("idx"), &JsValue::from(self.idx))
                        .unwrap();
                    Reflect::set(&result, &JsValue::from_str("ms"), &JsValue::from(ms)).unwrap();
                    Reflect::set(&result, &JsValue::from_str("va"), &JsValue::from(gap)).unwrap();
                }
            }
            self.accumulated_samples = rest.to_vec();
            self.idx = self.idx.wrapping_add(1);
        }

        return JsValue::from(result);
    }
}

/// Run entry point for the main thread.
#[wasm_bindgen]
pub fn startup() -> Worker {
    let worker_handle = Worker::new("./worker.js").unwrap();
    worker_handle
}
