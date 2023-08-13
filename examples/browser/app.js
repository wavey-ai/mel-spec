const { startup } = wasm_bindgen;

const canvas = document.getElementById("canvas");
const canvasCtx = canvas.getContext("2d");

const canvasHeight = canvas.height;
const fftSize = 1024;
const hopSize = 160;
const samplingRate = 16000;
const nMels = 80;

function convertToFloat(grayscaleValue) {
  return grayscaleValue / 255;
}

function colorizeGrayscaleValue(value, colormapName, reverse) {
  const x = convertToFloat(value);

  const colorTuple = evaluate_cmap(x, colormapName, reverse);

  return colorTuple;
}

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

  const melSab = sharedbuffer(nMels, 64, Uint8ClampedArray);
  const melBuf = ringbuffer(melSab, nMels, 64, Uint8ClampedArray);
  const pcmSab = sharedbuffer(128, 64, Float32Array);
  const pcmBuf = ringbuffer(pcmSab, 128, 64, Float32Array);

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
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

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

      for (let col = 0; col < numColumns; col++) {
        const startIdx = col * nMels;
        const endIdx = Math.min(startIdx + nMels, frame.length);
        const columnData = frame.slice(startIdx, endIdx);

        const arr = new Uint8ClampedArray(nMels * 4);
        for (let i = 0; i < columnData.length; i++) {
          let val = columnData[i]; // Use the correct index

          let [r, g, b] = [];
          if (!vad) {
            [r, g, b] = colorizeGrayscaleValue(val, "plasma", true);
          } else {
            [r, g, b] = colorizeGrayscaleValue(val, "cividis", true);
          }

          arr[i * 4 + 0] = r; // R value
          arr[i * 4 + 1] = g; // G value
          arr[i * 4 + 2] = b; // B value
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
        ctx.fillStyle = "black";
      } else {
        ctx.fillStyle = "red";
      }

      ctx.fill();
      ctx.closePath();
    }
  };

  const updateIntervalMs = 10;
  function updateUI() {
    let frames = [];
    while (true) {
      const mel = melBuf.pop();
      if (!mel) {
        break;
      }

      let vad = mel && (mel[0] & 1) === 1;
      addFrame(mel, vad);
    }
  }
  const updateIntervalId = setInterval(updateUI, updateIntervalMs);
}

startButton.addEventListener("click", () => {
  const audioContext = new AudioContext({ sampleRate: 16_000 });
  startAudioProcessing(audioContext);
});
