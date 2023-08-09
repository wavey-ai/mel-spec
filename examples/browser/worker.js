importScripts("./dist/mel_spec_pipeline.js");
importScripts("./ringbuffer.js");

const { SpeechToMel } = wasm_bindgen;

const instance = wasm_bindgen("./dist/mel_spec_pipeline_bg.wasm");

async function init_wasm_in_worker() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.

  await instance;

  console.log('Initializing worker')

  self.onmessage = async (event) => {
    const opt = event.data.options;
    const mod = SpeechToMel.new(
      opt.sourceSamplingRate,
      opt.fftSize,
      opt.hopSize,
      opt.samplingRate,
      opt.nMels
    );
    console.log("init");
    const melBuf = ringbuffer(event.data.melSab, opt.nMels, 1024, Uint8ClampedArray);
    const pcmBuf = ringbuffer(event.data.pcmSab, 128, 1024, Float32Array);

    while (true) {
      let samples = pcmBuf.pop();
      if (samples) {
        const res = mod.add(samples);
        if (res.ok) {
          melBuf.push(res.frame);
        }
      }
    }
  };
}

init_wasm_in_worker();
