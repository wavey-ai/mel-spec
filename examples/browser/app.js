const { startup } = wasm_bindgen;

const canvas = document.getElementById("canvas");
const canvasCtx = canvas.getContext("2d");

const canvasHeight = canvas.height;
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
    await audioContext.audioWorklet.addModule("./dist/worklet.js");
  } catch (error) {
    console.error("Error loading audio worklet:", error);
  }

  const audioNode = new AudioWorkletNode(audioContext, "AudioSender");
  volume.connect(audioNode);
  audioNode.connect(audioContext.destination);

  const melSab = sharedbuffer(nMels, 1024, Uint8ClampedArray);
  const melBuf = ringbuffer(melSab, nMels, 1024, Uint8ClampedArray);
  const pcmSab = sharedbuffer(128, 1024, Float32Array);
  const pcmBuf = ringbuffer(pcmSab, 128, 1024, Float32Array);

  setTimeout(() => {
    worker.postMessage({
      options: {
        sourceSamplingRate,
        fftSize,
        hopSize,
        samplingRate,
        nMels,
      },
      melSab,
      pcmSab,
    });
    audioNode.port.postMessage({
      pcmSab,
    });
  }, 500);

  const canvas = document.getElementById("canvas"); // Replace 'canvas' with the ID of your canvas element
  const ctx = canvas.getContext("2d");
  const columnWidth = 1;
  const canvasHeight = 130;
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

  let addFrame = (frame, vad) => {
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
          let val = columnData[columnData.length - i];
          arr[i * 4 + 0] = val; // R value
          arr[i * 4 + 1] = 50; // G value
          arr[i * 4 + 2] = val; // B value
          arr[i * 4 + 3] = 255; // A value
        }

        const imageData = new ImageData(arr, 1);
        ctx.putImageData(imageData, canvas.width - numColumns + col, 0);
      }

      // Draw circle based on vad flag
      const centerX = Math.floor(canvas.width / 2);
      const centerY = 100;
      const circleRadius = 10; // Adjust the radius as needed

      ctx.beginPath();
      ctx.arc(centerX, centerY, circleRadius, 0, 2 * Math.PI);

      if (vad) {
        // Draw a green circle when vad is true
        ctx.fillStyle = "red";
      } else {
        // Draw a red circle when vad is false
        ctx.fillStyle = "blue";
      }

      ctx.fill();
      ctx.closePath();
    }
  };

  let i = 0;
  let vad = false;
  const updateIntervalMs = 5;
  function updateUI() {
    const mel = melBuf.pop();
    if (vad) {
      i++
    }
    if (vad && i==100) {
        vad = false;
        i = 0;
    }
    if (!vad) {
      vad = mel && (mel[0] & 1) === 1;
    }
    addFrame(mel, vad);
  }
  const updateIntervalId = setInterval(updateUI, updateIntervalMs);
}

startButton.addEventListener("click", () => {
  const audioContext = new AudioContext({ sampleRate: 16_000 });
  startAudioProcessing(audioContext);
});
