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
  size: nMels + 8,
  max: 64,
};

const micBufOpts = {
  size: 128,
  max: 64,
};

const wavBufOpts = {
  size: 160,
  max: 200_000,
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

document.addEventListener("DOMContentLoaded", async function() {
  await startWorker();
  startUi();

  const form = document.getElementById("uploadForm");
  const fileInput = document.getElementById("waveFileInput");

  form.addEventListener("submit", async function(event) {
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

    reader.onload = function(event) {
      const chunk = event.target.result;

      if (chunk.byteLength > 0) {
        wav_worker.postMessage({ buf: chunk });
        offset += chunk.byteLength;
        readNextChunk();
      } else {
        console.log("Finished reading file.");
      }
    };

    reader.onerror = function() {
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

let x = 0;

function startUi() {
  const updateIntervalMs = 10;

  const melImages = [];
  let frames = [];
  let vads = 0;
  let min_len = 150;
  const accumulateFrame = (frame) => {
    frames.push(frame);
    if (!frame.vad) {
      if (frames.length >= min_len) {
        const dequant = frames.map((a) => a.toF32());
        let f = normMel(interleave(dequant));
        const tga = createTGAImage(f, nMels);
        newSegment(interleave(frames.map((a) => a.luma)), tga);
        frames = [];
      }
    }
  };

  const mels = document.getElementById("mels");

  const newSegment = async (frames, tga) => {
    const numColumns = frames.length / nMels;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const imageDataArr = new Uint8ClampedArray(nMels * 4 * numColumns);

    for (let col = 0; col < numColumns; col++) {
      for (let row = 0; row < nMels; row++) {
        const dataIndex = row * numColumns + col;
        const val = frames[dataIndex];
        const [r, g, b] = colorizeGrayscaleValue(val, "winter", false);
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
    spanElement.textContent = "Pending"; // Display "Pending" initially

    liElement.appendChild(imgElement);
    liElement.appendChild(spanElement);

    mels.appendChild(liElement);

    // Perform an asynchronous POST request to localhost:9000
    try {
      const response = await fetch("http://localhost:9000", {
        method: "POST",
        body: tga.buffer,
      });

      if (response.ok) {
        const textResponse = await response.text();
        spanElement.textContent = textResponse;
      } else {
        spanElement.textContent = "Error: Unable to fetch data.";
      }
    } catch (error) {
      spanElement.textContent = "Error: " + error.message;
    }
  };

  const updateUI = () => {
    while (true) {
      const mel = melBuf.pop();
      if (!mel) {
        break;
      }

      let frame = melFrame(mel);

      addFrame(frame.luma, frame.vad);
      accumulateFrame(frame);
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

function melFrame(mel) {
  const vad = !(mel && (mel[0] & 1) === 1);
  const luma = mel.slice(0, 80);
  luma.reverse();

  const minBytes = mel.slice(80, 84);
  const maxBytes = mel.slice(84, 88);

  const minArrayBuffer = new Uint8Array(minBytes).buffer;
  const maxArrayBuffer = new Uint8Array(maxBytes).buffer;

  const minDataView = new DataView(minArrayBuffer);
  const maxDataView = new DataView(maxArrayBuffer);

  const min = minDataView.getFloat32(0, true);
  const max = maxDataView.getFloat32(0, true);
  const toF32 = () => dequantize(mel.slice(0, 80), { min, max });

  return {
    luma,
    range: { min, max },
    vad,
    toF32,
  };
}
startButton.addEventListener("click", () => {
  const audioContext = new AudioContext({ sampleRate: 16_000 });
  startAudioProcessing(audioContext);
});

function createTGAImage(frames, n_mels) {
  const { data, range } = quantize(frames);
  let width = Math.floor(data.length / n_mels);
  let height = n_mels;

  // Combine the TGA header and image data
  let tga_header = new Uint8Array(26); // Increased length to accommodate range data
  tga_header[0] = 8; // ID len
  tga_header[1] = 0; // color map type (unused)
  tga_header[2] = 3; // Uncompressed, black and white images.
  tga_header.set(new Uint8Array(5), 3); // color map spec (unused)
  tga_header.set(new Uint8Array(4), 8); // X and Y Origin (unused)
  tga_header.set(new Uint8Array(new Uint16Array([width]).buffer), 12); // Image Width (little-endian)
  tga_header.set(new Uint8Array(new Uint16Array([height]).buffer), 14); // Image Height (little-endian)
  tga_header[16] = 8; // Bits per Pixel (8 bits)
  tga_header[17] = 0;

  // Store range data
  const rangeBuffer = new ArrayBuffer(8); // 2 floats, each 4 bytes
  const rangeView = new DataView(rangeBuffer);
  rangeView.setFloat32(0, range.min, true); // true indicates little-endian
  rangeView.setFloat32(4, range.max, true);
  tga_header.set(new Uint8Array(rangeBuffer), 18);

  let tga_image = new Uint8Array(tga_header.length + data.length);
  tga_image.set(tga_header);
  tga_image.set(data, tga_header.length);

  return tga_image;
}

function quantize(frame) {
  let result = new Uint8Array(frame.length);
  let min = Math.min(...frame);
  let max = Math.max(...frame);
  let scale = 255.0 / (max - min);
  for (let i = 0; i < frame.length; i++) {
    let scaled_value = Math.round((frame[i] - min) * scale);
    result[i] = scaled_value;
  }

  return { data: result, range: { min, max } };
}

// Dequantize Uint8Array data to original f32 values.
function dequantize(data, range) {
  let result = [];

  let scale = (range.max - range.min) / 255.0;

  for (let value of data) {
    let scaled_value = value * scale + range.min;
    result.push(scaled_value);
  }

  return result;
}

function downloadBytesAsFile(bytes, fileName) {
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.textContent = "Download TGA Bytes";

  document.body.appendChild(link);

  link.click();

  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function normMel(frame) {
  const mmax = frame.reduce((acc, x) => Math.max(acc, x), -Infinity);
  const clamped = frame.map((x) => (Math.min(x, mmax) + 4.0) / 4.0);

  return clamped;
}
