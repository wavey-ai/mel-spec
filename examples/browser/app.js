const { startup } = wasm_bindgen;

const canvas = document.getElementById("canvas");
const canvasCtx = canvas.getContext("2d");

const canvasHeight = canvas.height;
const pixelsPerColumn = 2;
const fftSize = 1024;
const hopSize = 160;
const samplingRate = 16000;
const nMels = 80;

async function startAudioProcessing(audioContext) {
  await wasm_bindgen();
  let worker = startup();

  const audioStream = await navigator.mediaDevices.getUserMedia({
    audio: true,
  });
  const sourceSamplingRate = audioContext.sampleRate;

  const volume = audioContext.createGain();
  const audioInput = audioContext.createMediaStreamSource(audioStream);
  audioInput.connect(volume);

  try {
    await audioContext.audioWorklet.addModule("audioSender.js");
  } catch (error) {
    console.error("Error loading audioSender.js:", error);
  }
  const audioNode = new AudioWorkletNode(audioContext, "AudioSender");
  volume.connect(audioNode);
  audioNode.connect(audioContext.destination);

  let bufferSize = nMels * 1024 + 2 + 1;
  const sab = new SharedArrayBuffer(bufferSize); // One extra byte for the length information
  worker.postMessage({
    type: "init",
    options: {
      sourceSamplingRate,
      fftSize,
      hopSize,
      samplingRate,
      nMels,
      melBuffer: sab,
    },
  });

  const canvas = document.getElementById("canvas"); // Replace 'canvas' with the ID of your canvas element
  const ctx = canvas.getContext("2d");
  const columnWidth = 1;
  const canvasHeight = 80;
  ctx.fillStyle = "rgb(0, 0, 0)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  const arr = new Uint8ClampedArray((nMels + 8) * 4);
  for (let i = 0; i < arr.length; i += 4) {
    arr[i + 0] = 255; // R value
    arr[i + 1] = 255; // G value
    arr[i + 2] = 255; // B value
    arr[i + 3] = 255; // A value
  }

  const boundary = new ImageData(arr, 1);

  let addFrame = (frame) => {
    if (frame && frame.length > 0) {
      const numColumns = Math.ceil(frame.length / nMels);

      // Shift the pixels left by numColumns columns using global composite operation "copy"
      ctx.globalCompositeOperation = "copy";
      ctx.drawImage(
        canvas,
        numColumns * columnWidth,
        0,
        canvas.width - numColumns * columnWidth,
        nMels + 8,
        0,
        0,
        canvas.width - numColumns * columnWidth,
        nMels + 8
      );
      ctx.globalCompositeOperation = "source-over";

      // Draw each column on the right side of the canvas
      for (let col = 0; col < numColumns; col++) {
        const startIdx = col * nMels;
        const endIdx = Math.min(startIdx + nMels, frame.length);
        const columnData = frame.slice(startIdx, endIdx);

        const arr = new Uint8ClampedArray(nMels * 4);
        for (let i = 0; i < columnData.length; i++) {
          let val = columnData[columnData.length-i];
          arr[i * 4 + 0] = val; // R value
          arr[i * 4 + 1] = 50; // G value
          arr[i * 4 + 2] = val; // B value
          arr[i * 4 + 3] = 255; // A value
        }

        const imageData = new ImageData(arr, 1);
        ctx.putImageData(imageData, canvas.width - numColumns + col, 0);
      }
    }
  };

  let i = 0;
  audioNode.port.onmessage = (event) => {
    let samples = event.data.samples;
    worker.postMessage({ samples }, [samples.buffer]);

    if (i % 4 == 0) {
      const lockVariable = new Int32Array(sab, 0, 4);

      // Acquire the lock
      if (Atomics.compareExchange(lockVariable, 0, 1) === 0) {
        let melMemory = new Uint8Array(sab);
        let cursor = new DataView(melMemory.buffer, 4).getInt16(0, true);
        const offset = 6 + (cursor * nMels);
        if (cursor > 0) {
          const frames = melMemory.slice(6, offset);
          // Zero out the data portion of the SharedArrayBuffer (excluding the lock variable)
          const dataView = new Uint8Array(sab, 4); // Start at offset 4 to skip the lock variable
          dataView.fill(0);
          Atomics.store(lockVariable, 0, 0);
          Atomics.notify(lockVariable, 0, 1);

          addFrame(frames);
        }
      }
    }
    i += 1;

  };
}

startButton.addEventListener("click", () => {
  const audioContext = new AudioContext({ sampleRate: 16_000 });
  startAudioProcessing(audioContext);
});
