importScripts("./dist/soundkit.js");
importScripts("./ringbuffer.js");

const { WavToPcm } = wasm_bindgen_wav;

const instance = wasm_bindgen_wav("./dist/soundkit_bg.wasm");

async function init_wasm_in_worker() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
  await instance;

  const mod = WavToPcm.new();
  let len = null;
  let pcmBuf = null;

  self.onmessage = async (event) => {
    if (event.data.pcmSab) {
      let opts = event.data;
      len = opts.pcmBufOpts.size,
        pcmBuf = ringbuffer(
          opts.pcmSab,
          opts.pcmBufOpts.size,
          opts.pcmBufOpts.max,
          Float32Array,
        );
    } else {
      var byteArray = new Int8Array(event.data.buf);
      let audio = mod.add(byteArray);
      if (audio.ok && audio.channels) {
        console.log(audio);
        let res = new Float32Array(len);
        let j = 0;
        for (let i = 0; i < audio.channels[0].length; i++) {
          if (i > 0 && i % len === 0) {
            pcmBuf.push(res);
            res = new Float32Array(len);
            j = 0;
          }
          res[j] = audio.channels[0][i];
          j++;
        }
      }
    }
  };
}

init_wasm_in_worker();
