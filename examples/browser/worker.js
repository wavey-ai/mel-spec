importScripts("./dist/mel_spec_pipeline.js");

const { SpeechToMel } = wasm_bindgen;

let mod = null;
let sab = null;
let nMels = 0;
const instance = wasm_bindgen("./dist/mel_spec_pipeline_bg.wasm");

async function init_wasm_in_worker() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
  await instance;

  self.onmessage = async (event) => {
    if (event.data.options) {
      let opt = event.data.options;
      sab = new Uint8Array(event.data.options.melBuffer);
      mod = SpeechToMel.new(
        opt.sourceSamplingRate,
        opt.fftSize,
        opt.hopSize,
        opt.samplingRate,
        opt.nMels
      );
      nMels = opt.nMels;
    } else if (mod && event.data.samples) {
      let lockVariable = new Int32Array(sab, 0, 1);
      while (Atomics.compareExchange(lockVariable, 0, 1) !== 0) {
        Atomics.wait(lockVariable, 0, 1);
      }

      let data = event.data.samples;
      let res = mod.add(data);

      if (res.frame && res.frame.length > 0) {
        let cursor = new DataView(sab.buffer, 4).getInt16(0, true);
        const offset = 4 + 2 + (cursor * nMels);
        sab.set(res.frame, offset); // Copy frame data after the cursor
        new DataView(sab.buffer, 4).setInt16(0, cursor + 1, true);
      }

      Atomics.store(lockVariable, 0, 0);
      Atomics.notify(lockVariable, 0, 1);
    }
  };
}

init_wasm_in_worker();
