registerProcessor(
  "AudioSender",
  class AudioSender extends AudioWorkletProcessor {
    constructor() {
      super();
    }

    process(inputs, outputs) {
      let samples = inputs[0][0];
      this.port.postMessage({ samples }, [samples.buffer]);
      return true;
    }
  }
);
