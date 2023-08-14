importScripts("./dist/mel_spec_pipeline.js");
importScripts("./ringbuffer.js");

const { SpeechToMel } = wasm_bindgen;

const instance = wasm_bindgen("./dist/mel_spec_pipeline_bg.wasm");

async function init_wasm_in_worker() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.

  await instance;
  let melBuf;
  let pcmBuf;
  let mod;

  self.onmessage = async (event) => {
    const opts = event.data;
    if (opts.melBufOpts) {
      mod = SpeechToMel.new(
        opts.fftSize,
        opts.hopSize,
        opts.samplingRate,
        opts.nMels
      );
      melBuf = ringbuffer(
        opts.melSab,
        opts.melBufOpts.size,
        opts.melBufOpts.max,
        Uint8ClampedArray,
      );
    }

    if (opts.pcmBufOpts) {
      pcmBuf = ringbuffer(
        opts.pcmSab,
        opts.pcmBufOpts.size,
        opts.pcmBufOpts.max,
        Float32Array,
      );
    }

    if (opts.pop && pcmBuf) {
      while (true) {
        let samples = pcmBuf.pop();
        if (samples) {
          const res = mod.add(samples);
          if (res.ok) {
            let f = res.frame;
            f[0] = res.va ? f[0] & ~1 : f[0] | 1;
            melBuf.push(f);
          }
        } else {
          break;
        }
      }
    }
  };
}

init_wasm_in_worker();
