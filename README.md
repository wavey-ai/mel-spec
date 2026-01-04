# Mel Spec

[![CI](https://github.com/wavey-ai/mel-spec/actions/workflows/ci.yml/badge.svg)](https://github.com/wavey-ai/mel-spec/actions/workflows/ci.yml)

A Rust implementation of mel spectrograms with support for:
- **Whisper-compatible** mel spectrograms (aligned with whisper.cpp, PyTorch, librosa)
- **Kaldi-compatible** filterbank features (matching kaldi_native_fbank output)
- **NeMo-compatible** mel filters

## Examples

See [wavey-ai/hush](https://github.com/wavey-ai/hush) for live demo

![image](doc/browser.png)
* [stream microphone or wav to mel wasm worker](examples/browser)
* [stream from ffmpeg to whisper.cpp](examples/stream_whisper)
* [convert audio to mel spectrograms and save to image](examples/mel_tga)
* [transcribe images with whisper.cpp](examples/tga_whisper)

## Usage

```rust
use mel_spec::prelude::*;
```

### Kaldi-compatible Filterbank Features

The `fbank` module provides Kaldi-style filterbank features with parity to `kaldi_native_fbank`. This is useful for speaker embedding models like WeSpeaker and pyannote.

```rust
use mel_spec::fbank::{Fbank, FbankConfig};

// Default config matches kaldi_native_fbank defaults
let config = FbankConfig::default();
let fbank = Fbank::new(config);

// Compute features from audio samples (mono, f32, 16kHz)
let features = fbank.compute(&samples);
// Returns Array2<f32> with shape (num_frames, 80)
```

**Kaldi defaults:**
- Sample rate: 16000 Hz
- Mel bins: 80
- Frame length: 25ms (400 samples)
- Frame shift: 10ms (160 samples)
- Window: Povey (like Hamming but goes to zero at edges)
- Preemphasis: 0.97
- CMN: enabled (subtract mean across time)

### Mel Filterbank (Whisper/librosa compatible)

Mel filterbanks within 1.0e-7 of librosa and identical to whisper GGML model-embedded filters.

```rust
use mel_spec::mel::mel;

let sampling_rate = 16000.0;
let fft_size = 400;
let n_mels = 80;
let filters = mel(sampling_rate, fft_size, n_mels, None, None, false, true);
// Returns Array2<f64> with shape (80, 201)
```

### Spectrogram using Short Time Fourier Transform

STFT with overlap-and-save that has parity with PyTorch and whisper.cpp. Suitable for streaming audio.

```rust
use mel_spec::stft::Spectrogram;

let fft_size = 400;
let hop_size = 160;
let mut spectrogram = Spectrogram::new(fft_size, hop_size);

// Add PCM audio samples
let samples: Vec<f32> = vec![0.0; 1600];
if let Some(fft_frame) = spectrogram.add(&samples) {
    // Use FFT result
}
```

### STFT to Mel Spectrogram

Apply a pre-computed filterbank to FFT results. Output is identical to whisper.cpp and whisper.py.

```rust
use mel_spec::mel::MelSpectrogram;
use ndarray::Array1;
use num::Complex;

let fft_size = 400;
let sampling_rate = 16000.0;
let n_mels = 80;
let mut mel_spec = MelSpectrogram::new(fft_size, sampling_rate, n_mels);

let fft_input = Array1::from(vec![Complex::new(1.0, 0.0); fft_size]);
let mel_frame = mel_spec.add(fft_input);
```

### RingBuffer for Streaming

For creating spectrograms from streaming audio, see `RingBuffer` in [rb.rs](src/rb.rs).

### Saving Mel Spectrograms to TGA

Mel spectrograms can be saved as 8-bit TGA images (uncompressed, supported by macOS and Windows). These images encode quantized mel spectrogram data that whisper.cpp can process directly without audio input.

```rust
use mel_spec::quant::{save_tga_8bit, load_tga_8bit};

// Save spectrogram
save_tga_8bit(&mel_data, "spectrogram.tga").unwrap();

// Load and use with whisper.cpp
let mel = load_tga_8bit("spectrogram.tga").unwrap();
```

TGA files are lossless for speech-to-text - they encode all information available in the model's view of raw audio.

```
ffmpeg -i audio.mp3 -f f32le -ar 16000 -ac 1 pipe:1 | ./target/release/tga_whisper -t spectrogram.tga
```

![image](doc/cutsec_46997.png)
_"the quest for peace."_

### Voice Activity Detection

Uses Sobel edge detection to find speech boundaries in mel spectrograms. This enables real-time processing by finding natural cut points between words/phrases.

```rust
use mel_spec::vad::{VoiceActivityDetector, DetectionSettings};

let settings = DetectionSettings {
    min_energy: 1.0,
    min_y: 3,
    min_x: 5,
    min_mel: 0,
    min_frames: 100,
};
let vad = VoiceActivityDetector::new(settings);
```

Speech in mel spectrograms is characterized by clear gradients. The VAD finds vertical gaps suitable for cutting, and drops frames that look like gaps in speech (which cause Whisper hallucinations).

![image](doc/jfk_vad_example.png)

**Examples from JFK speech:**

Energy but no speech - VAD correctly rejects:
![image](doc/frame_23760.png)
![image](testdata/vad_off_23760.png)

Fleeting word - VAD correctly detects:
![image](doc/frame_27125.png)
![image](testdata/vad_on_27125.png)

Full JFK transcript with VAD: [jfk_transcript_golden.txt](doc/jfk_transcript_golden.txt)

## Performance

### CPU Performance

Benchmarks on Apple M1 Pro (single-threaded, release build):

| Audio Length | Frames | Time | Throughput |
|-------------|--------|------|------------|
| 10s | 997 | 21ms | 476x realtime |
| 60s | 5997 | 124ms | 484x realtime |
| 300s (5 min) | 29997 | 622ms | 482x realtime |

Full mel spectrogram pipeline (STFT + mel filterbank + log) at 16kHz, FFT size 512, hop 160, 80 mel bins.

**CPU performance is excellent** - processing 5 minutes of audio in 622ms means the library is ~480x faster than realtime on a single core.

### GPU Acceleration

This library focuses on pure Rust CPU implementation. For GPU acceleration, consider:

| Option | Speedup | Notes |
|--------|---------|-------|
| **NVIDIA NeMo** | ~10x over CPU | Python/PyTorch, uses cuBLAS/cuDNN, best for batch processing |
| **torchaudio** | ~5-10x | Python/PyTorch, CUDA backend |
| **mel-spec gpu branch** | ~1.6x | Experimental, requires CUDA toolkit + nvcc |

**Options:**

1. **NeMo / torchaudio** → Python/PyTorch with CUDA, best for batch processing
2. **gpu branch** → Experimental CUDA support (~1.6x speedup), requires CUDA toolkit + nvcc
3. **wgpu/rust-gpu** → Pure Rust GPU (ecosystem maturing)

The `gpu` branch keeps the full pipeline on GPU (STFT → mel filterbank → log). Requires C++ CUDA kernels and NVIDIA hardware.

## Discussion

* Mel spectrograms encode at 6.4KB/sec (80 x 2 bytes x 40 frames)
* Float PCM for Whisper is 64KB/sec at 16kHz

whisper.cpp produces mel spectrograms with 1.0e-6 precision, but these are invariant to 8-bit quantization. We can save as 8-bit images without losing useful information.

**We only need 1.0e-1 precision for accurate results**, and rounding may actually improve some difficult transcriptions:

```
Original:  [0.158, 0.266, 0.076, 0.196, 0.167, ...]
Rounded:   [0.2,   0.3,   0.1,   0.2,   0.2,   ...]
```

Once quantized, the spectrograms are the same:

![image](./doc/quantized_mel.png)
![image](./doc/quantized_mel_e1.png)
_(top: original, bottom: rounded to 1.0e-1)_

Speech is encapsulated almost entirely in the frequency domain, and the mel scale effectively divides frequencies into 80 bins. 8-bits of grayscale is probably overkill - it could be compressed further.
