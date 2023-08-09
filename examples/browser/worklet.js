let buf = null;

registerProcessor(
  "AudioSender",
  class AudioSender extends AudioWorkletProcessor {
    constructor() {
      super();

      this.port.onmessage = async (event) => {
        buf = ringbuffer(event.data.pcmSab, 128, 1024, Float32Array);
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
