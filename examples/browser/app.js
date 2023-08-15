const { startup } = wasm_bindgen;
const { startup: startup_wav } = wasm_bindgen_wav;

const canvas = document.getElementById("canvas");
const canvasCtx = canvas.getContext("2d");

const canvasHeight = canvas.height;
const fftSize = 1024;
const hopSize = 160;
const samplingRate = 16000;
const nMels = 80;

const melBufOpts = {
  size: nMels,
  max: 64,
};

const micBufOpts = {
  size: 128,
  max: 64,
};

const wavBufOpts = {
  size: 160,
  max: 50_000,
};

const melSab = sharedbuffer(melBufOpts.size, melBufOpts.max, Uint8ClampedArray);
const melBuf = ringbuffer(
  melSab,
  melBufOpts.size,
  melBufOpts.max,
  Uint8ClampedArray
);
const micSab = sharedbuffer(micBufOpts.size, micBufOpts.max, Float32Array);
const wavSab = sharedbuffer(wavBufOpts.size, wavBufOpts.max, Float32Array);

let wav_worker;

function convertToFloat(grayscaleValue) {
  return grayscaleValue / 255;
}

function colorizeGrayscaleValue(value, colormapName, reverse) {
  const x = convertToFloat(value);

  const colorTuple = evaluate_cmap(x, colormapName, reverse);

  return colorTuple;
}

let addFrame;

document.addEventListener("DOMContentLoaded", async function () {
  await startWorker();
  startUi();

  const form = document.getElementById("uploadForm");
  const fileInput = document.getElementById("waveFileInput");

  form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
      alert("Please select a WAV file.");
      return;
    }

    pcm_worker.postMessage({ pcmSab: wavSab, pcmBufOpts: wavBufOpts });
    const CHUNK_SIZE = 1024 * 1024;
    let offset = 0;

    const reader = new FileReader();

    reader.onload = function (event) {
      const chunk = event.target.result;

      if (chunk.byteLength > 0) {
        wav_worker.postMessage({ buf: chunk });
        offset += chunk.byteLength;
        readNextChunk();
      } else {
        console.log("Finished reading file.");
      }
    };

    reader.onerror = function () {
      console.error("Error reading file.");
    };

    function readNextChunk() {
      if (offset < file.size) {
        const fileSlice = file.slice(offset, offset + CHUNK_SIZE);
        reader.readAsArrayBuffer(fileSlice);
      }
    }

    readNextChunk();
  });

  const canvas = document.getElementById("canvas"); // Replace 'canvas' with the ID of your canvas element
  const ctx = canvas.getContext("2d");
  const columnWidth = 1;
  const canvasHeight = 130;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  addFrame = (frame, vad) => {
    if (frame && frame.length > 0) {
      const numColumns = Math.ceil(frame.length / nMels);

      // Shift the pixels left by numColumns columns using global composite operation "copy"
      ctx.globalCompositeOperation = "copy";
      ctx.drawImage(
        canvas,
        numColumns * columnWidth,
        0,
        canvas.width - numColumns * columnWidth,
        nMels + 16,
        0,
        0,
        canvas.width - numColumns * columnWidth,
        nMels + 16
      );
      ctx.globalCompositeOperation = "source-over";

      for (let col = 0; col < numColumns; col++) {
        const startIdx = col * nMels;
        const endIdx = Math.min(startIdx + nMels, frame.length);
        const columnData = frame.slice(startIdx, endIdx);

        let arr = new Uint8ClampedArray(nMels * 4);
        for (let i = 0; i < columnData.length; i++) {
          let val = columnData[i]; // Use the correct index

          let [r, g, b] = [];
          if (vad) {
            [r, g, b] = colorizeGrayscaleValue(val, "plasma", false);
          } else {
            [r, g, b] = colorizeGrayscaleValue(val, "cividis", false);
          }

          arr[i * 4 + 0] = r; // R value
          arr[i * 4 + 1] = g; // G value
          arr[i * 4 + 2] = b; // B value
          arr[i * 4 + 3] = 255; // A value
        }

        for (let i = 0; i < 2 * (vad ? 6 : 4); i++) {
          arr = new Uint8ClampedArray([...arr, 0, 0, 0, 0]);
        }

        const [pixelR, pixelG, pixelB] = vad ? [255, 0, 0] : [0, 0, 0];
        let yOffset = vad ? 8 : 0; // Offset for vad true condition
        for (let i = 0; i < 4; i++) {
          arr[arr.length - 4 * (4 - i)] = pixelR; // R value
          arr[arr.length - 4 * (4 - i) + 1] = pixelG; // G value
          arr[arr.length - 4 * (4 - i) + 2] = pixelB; // B value
          arr[arr.length - 4 * (4 - i) + 3] = 255; // A value
        }

        const imageData = new ImageData(arr, 1);
        ctx.putImageData(imageData, canvas.width - numColumns + col, 0);
      }
      // Draw circle based on vad flag
      const centerX = 990;
      const centerY = 110;
      const circleRadius = 10; // Adjust the radius as needed

      ctx.beginPath();
      ctx.arc(centerX, centerY, circleRadius, 0, 2 * Math.PI);

      if (vad) {
        ctx.fillStyle = "red";
      } else {
        ctx.fillStyle = "black";
      }

      ctx.fill();
      ctx.closePath();
    }
  };
});

async function startWorker() {
  await wasm_bindgen();
  await wasm_bindgen_wav();

  pcm_worker = startup("./worker.js");
  wav_worker = startup_wav("./wav_worker.js");

  setTimeout(() => {
    wav_worker.postMessage({ pcmSab: wavSab, pcmBufOpts: wavBufOpts });
    pcm_worker.postMessage({
      fftSize,
      hopSize,
      samplingRate,
      nMels,
      melSab,
      melBufOpts,
    });
  }, 500);

  const updateIntervalMs = 10;

  const pop = () => {
    pcm_worker.postMessage({ pop: true });
  };
  const popIntervalId = setInterval(pop, updateIntervalMs);
}

function interleave(columns) {
  const numRows = columns[0].length;
  const numColumns = columns.length;
  const interleavedArray = new Array(numRows * numColumns);

  for (let row = 0; row < numRows; row++) {
    for (let col = 0; col < numColumns; col++) {
      interleavedArray[row * numColumns + col] = columns[col][row];
    }
  }

  return interleavedArray;
}

function startUi() {
  const updateIntervalMs = 10;

  const melImages = [];
  let segments = [];
  let frames = [];
  let vadCounter = 0; // Counter for contiguous !vad conditions
  let vads = 0
  const accumulateFrame = (mel, vad) => {
    if (!vad) {
      vadCounter++;
      if (vadCounter >= 10 || segments.length >= 100) {
        segments = segments.concat(frames);
        if (segments.length >= 20 && vads >= segments.length / 2) {
          newSegment(interleave(segments));
        }
        segments = [];
        frames = [];
        vadCounter = 0;
        vads = 0;
      } else if (segments.length > 0) {
        frames.push(mel);
      }
    } else {
      vadCounter = 0;
      vads += 1;
      frames.push(mel);
      if (frames.length === 5) {
        segments = segments.concat(frames);
        frames = [];
      }
    }
  };

  const mels = document.getElementById("mels");

  const newSegment = (frames) => {
    const numColumns = frames.length / nMels;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const imageDataArr = new Uint8ClampedArray(nMels * 4 * numColumns);

    for (let col = 0; col < numColumns; col++) {
      for (let row = 0; row < nMels; row++) {
        const dataIndex = row * numColumns + col;
        const val = frames[dataIndex];
        [r, g, b] = colorizeGrayscaleValue(val, "winter", false);
        imageDataArr[(row * numColumns + col) * 4 + 0] = r; // R value
        imageDataArr[(row * numColumns + col) * 4 + 1] = g; // G value
        imageDataArr[(row * numColumns + col) * 4 + 2] = b; // B value
        imageDataArr[(row * numColumns + col) * 4 + 3] = 255; // A value
      }
    }

    const imageData = new ImageData(imageDataArr, numColumns, nMels);
    canvas.width = numColumns;
    canvas.height = nMels;
    ctx.putImageData(imageData, 0, 0);

    const imageURI = canvas.toDataURL();

    const liElement = document.createElement("li");

    const imgElement = document.createElement("img");
    imgElement.src = imageURI;

    const spanElement = document.createElement("span");
    // You can add content to the span element here if needed

    liElement.appendChild(imgElement);
    liElement.appendChild(spanElement);

    mels.appendChild(liElement);
  };

  const updateUI = () => {
    while (true) {
      const mel = melBuf.pop();
      if (!mel) {
        break;
      }

      let vad = !(mel && (mel[0] & 1) === 1);
      mel.reverse();
      addFrame(mel, vad);
      accumulateFrame(mel, vad);
    }
  };

  const updateIntervalId = setInterval(updateUI, updateIntervalMs);
}

async function startAudioProcessing(audioContext) {
  audioStream = await navigator.mediaDevices.getUserMedia({
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

  pcm_worker.postMessage({ pcmSab: micSab, pcmBufOpts: micBufOpts });

  setTimeout(() => {
    audioNode.port.postMessage({
      pcmSab: micSab,
      pcmBufOpts: micBufOpts,
    });
  }, 500);
}

startButton.addEventListener("click", () => {
  const audioContext = new AudioContext({ sampleRate: 16_000 });
  startAudioProcessing(audioContext);
});
