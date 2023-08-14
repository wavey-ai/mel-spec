let buf = null;

registerProcessor(
  "AudioSender",
  class AudioSender extends AudioWorkletProcessor {
    constructor() {
      super();

      this.port.onmessage = async (event) => {
        let opts = event.data;
        buf = ringbuffer(
          opts.pcmSab,
          opts.pcmBufOpts.size,
          opts.pcmBufOpts.max,
          Float32Array,
        );
      };
    }

    process(inputs, outputs) {
      if (buf) {
        let samples = inputs[0][0];
        buf.push(samples);
      }
      return true;
    }
  }
);
