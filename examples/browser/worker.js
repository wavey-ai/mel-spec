importScripts("./dist/mel_spec.js");
importScripts("./ringbuffer.js");

const { SpeechToMel } = wasm_bindgen;

const instance = wasm_bindgen("./dist/mel_spec_bg.wasm");

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
          const res = mod.add(samples, true); // true for voice activity detection
          if (res.ok) {
            const data = new Uint8ClampedArray(res.frame.length + 8);
            data.set(res.frame);
            const float1Bytes = new Uint8Array(new Float32Array([res.min]).buffer);
            const float2Bytes = new Uint8Array(new Float32Array([res.max]).buffer);
            data.set(float1Bytes, 80);
            data.set(float2Bytes, 84);
            data[0] = res.va ? data[0] & ~1 : data[0] | 1;
            melBuf.push(data);
          }
        } else {
          break;
        }
      }
    }
  };
}

init_wasm_in_worker();
