use crate::mel::{norm_mel_in_place_f64, whisper_power_spectrum, SparseMelFilterbank};
use crate::quant::quantize;
use crate::stft::Spectrogram;
use crate::vad::{duration_ms_for_n_frames, DetectionSettings, VoiceActivityDetector};
use js_sys::{Object, Reflect, Uint8Array, Uint8ClampedArray};
use ndarray::Array2;
use wasm_bindgen::prelude::*;
use web_sys::Worker;

#[wasm_bindgen]
pub struct SpeechToMel {
    mel: SparseMelFilterbank,
    mel_vad: SparseMelFilterbank,
    power: Vec<f64>,
    frame: Vec<f64>,
    frame_vad: Vec<f64>,
    fft: Spectrogram,
    vad: VoiceActivityDetector,
    fft_size: usize,
    hop_size: usize,
    sampling_rate: f64,
    accumulated_samples: Vec<f32>,
    idx: usize,
}

#[wasm_bindgen]
impl SpeechToMel {
    #[wasm_bindgen]
    pub fn new(fft_size: usize, hop_size: usize, sampling_rate: f64, n_mels: usize) -> Self {
        Self::new_with_settings(
            fft_size,
            hop_size,
            sampling_rate,
            n_mels,
            DetectionSettings {
                min_energy: 1.0,
                min_y: 3,
                min_x: 3,
                min_mel: 0,
            },
        )
    }

    // The JavaScript constructor takes each setting as a positional argument.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = newWithVadSettings)]
    pub fn new_with_vad_settings(
        fft_size: usize,
        hop_size: usize,
        sampling_rate: f64,
        n_mels: usize,
        min_energy: f64,
        min_y: usize,
        min_x: usize,
        min_mel: usize,
    ) -> Self {
        Self::new_with_settings(
            fft_size,
            hop_size,
            sampling_rate,
            n_mels,
            DetectionSettings {
                min_energy,
                min_y,
                min_x,
                min_mel,
            },
        )
    }

    fn new_with_settings(
        fft_size: usize,
        hop_size: usize,
        sampling_rate: f64,
        n_mels: usize,
        settings: DetectionSettings,
    ) -> Self {
        let mel =
            SparseMelFilterbank::from_mel(sampling_rate, fft_size, n_mels, None, None, false, true);
        let mel_vad = SparseMelFilterbank::from_mel(
            sampling_rate,
            fft_size,
            n_mels / 4,
            None,
            None,
            false,
            true,
        );
        let stft = Spectrogram::new(fft_size, hop_size);
        let vad = VoiceActivityDetector::new(&settings);
        Self {
            accumulated_samples: Vec::new(),
            power: vec![0.0; mel.fft_bins()],
            frame: vec![0.0; mel.n_mels()],
            frame_vad: vec![0.0; mel_vad.n_mels()],
            mel,
            mel_vad,
            fft: stft,
            vad,
            sampling_rate,
            fft_size,
            hop_size,
            idx: 0,
        }
    }

    #[wasm_bindgen]
    pub fn get(&mut self) -> JsValue {
        let empty = vec![0.0; 0];
        self.add(empty, false)
    }

    #[wasm_bindgen]
    pub fn add(&mut self, data: Vec<f32>, vad: bool) -> JsValue {
        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(false)).unwrap();
        self.accumulated_samples.extend_from_slice(&data);
        if self.accumulated_samples.len() >= self.hop_size {
            let samples = &self.accumulated_samples[..self.hop_size];

            Reflect::set(
                &result,
                &JsValue::from_str("len"),
                &JsValue::from(samples.len()),
            )
            .unwrap();

            if let Some(spectrum) = self.fft.add_half(samples) {
                whisper_power_spectrum(spectrum, self.fft_size / 2, &mut self.power);
                self.mel.project_log10_f64(&self.power, &mut self.frame);
                let frame = self.frame.iter().map(|v| *v as f32).collect::<Vec<_>>();
                let (quant_frame, range) = quantize(&frame);
                let frame_array = Uint8Array::from(&quant_frame[..]);
                let frame_clamped_array = Uint8ClampedArray::new(&frame_array.buffer());
                Reflect::set(&result, &JsValue::from_str("frame"), &frame_clamped_array).unwrap();
                Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(true)).unwrap();
                Reflect::set(
                    &result,
                    &JsValue::from_str("min"),
                    &JsValue::from(range.min),
                )
                .unwrap();
                Reflect::set(
                    &result,
                    &JsValue::from_str("max"),
                    &JsValue::from(range.max),
                )
                .unwrap();
                let ms = duration_ms_for_n_frames(self.hop_size, self.sampling_rate, self.idx);
                Reflect::set(&result, &JsValue::from_str("idx"), &JsValue::from(self.idx)).unwrap();
                Reflect::set(&result, &JsValue::from_str("ms"), &JsValue::from(ms)).unwrap();
                if vad {
                    self.mel_vad
                        .project_log10_f64(&self.power, &mut self.frame_vad);
                    norm_mel_in_place_f64(&mut self.frame_vad);
                    let frame2 =
                        Array2::from_shape_vec((self.frame_vad.len(), 1), self.frame_vad.clone())
                            .expect("VAD frame shape matches the filterbank");
                    if let Some(gap) = self.vad.add(&frame2) {
                        Reflect::set(&result, &JsValue::from_str("va"), &JsValue::from(gap))
                            .unwrap();
                    }
                }
            }
            self.accumulated_samples.drain(..self.hop_size);
            self.idx = self.idx.wrapping_add(1);
        }

        JsValue::from(result)
    }
}

/// Run entry point for the main thread.
#[wasm_bindgen]
pub fn startup(path: String) -> Worker {
    Worker::new(&path).unwrap()
}
