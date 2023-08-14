use crate::wav::WavStreamProcessor;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;
use web_sys::Worker;

#[wasm_bindgen]
pub struct WavToPcm {
    wav: WavStreamProcessor,
}

#[wasm_bindgen]
impl WavToPcm {
    #[wasm_bindgen]
    pub fn new() -> Self {
        let wav = WavStreamProcessor::new();
        Self { wav }
    }

    #[wasm_bindgen]
    pub fn add(&mut self, data: &[u8]) -> JsValue {
        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(false)).unwrap();
        match self.wav.add(data) {
            Ok(Some(audio)) => {
                Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(true)).unwrap();
                Reflect::set(
                    &result,
                    &JsValue::from_str("bits_per_sample"),
                    &JsValue::from(audio.bits_per_sample()),
                )
                .unwrap();
                Reflect::set(
                    &result,
                    &JsValue::from_str("sampling_rate"),
                    &JsValue::from(audio.sampling_rate()),
                )
                .unwrap();
                Reflect::set(
                    &result,
                    &JsValue::from_str("channel_count"),
                    &JsValue::from(audio.channel_count()),
                )
                .unwrap();

                let js_array = audio
                    .channels()
                    .iter()
                    .map(|channel| {
                        channel
                            .iter()
                            .map(|&value| JsValue::from_f64(f64::from(value)))
                            .collect::<Array>()
                    })
                    .collect::<Array>();
                Reflect::set(&result, &JsValue::from_str("channels"), &js_array).unwrap();
            }
            Ok(None) => {
                return JsValue::from(result);
            }
            Err(error) => {
                // Handle the error
                println!("Error: {}", error);
            }
        }

        return JsValue::from(result);
    }
}

/// Run entry point for the main thread.
#[wasm_bindgen]
pub fn startup(path: String) -> Worker {
    let worker_handle = Worker::new(&path).unwrap();
    worker_handle
}
