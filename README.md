# Mel Spec

[![CI](https://github.com/wavey-ai/mel-spec/actions/workflows/ci.yml/badge.svg)](https://github.com/wavey-ai/mel-spec/actions/workflows/ci.yml)

Fast Rust mel spectrogram and VAD primitives for ASR systems.

`mel-spec` is built around the parts of speech pipelines that need to be cheap,
predictable, and easy to embed: STFT, Whisper-compatible log-mel features,
Kaldi-style filterbanks, TGA spectrogram interchange, and a lightweight VAD that
reuses the same mel/STFT features.

## Main Features

| Feature | What it is for |
| --- | --- |
| Whisper-compatible mel | Log-mel spectrograms aligned with whisper.cpp, PyTorch, and librosa. |
| Kaldi-compatible fbank | 80-bin Kaldi-style filterbank features for speaker and audio models. |
| Streaming STFT | Overlap-and-save STFT for live audio pipelines. |
| Model-free VAD | Fast speech/non-speech decisions and timestamps from mel spectrogram structure. |
| TGA mel images | Store and pass quantized mel spectrograms as simple 8-bit TGA files. |
| Local Whisper WASM | Hush uses `mel-spec` mel tensors/TGA segments for fully local browser Whisper transcription. |
| Native GPU backends | Experimental CUDA and `wgpu` paths for batched native mel generation. |
| Browser worker demo | WASM worker and SharedArrayBuffer example for live browser audio. |

## Quick Start

```rust
use mel_spec::prelude::*;

let samples = vec![0.0_f32; 16_000];
let mel_frames = Spectrogram::compute_mel_spectrogram_cpu(
    &samples,
    400,
    160,
    80,
    16_000.0,
);

println!("frames={}", mel_frames.len());
```

The focused API examples are kept in the example READMEs so the top-level
README stays readable:

| Example | Description |
| --- | --- |
| [browser](examples/browser) | Stream microphone or WAV audio to a WASM mel worker. |
| [mel_tga](examples/mel_tga) | Convert raw audio to TGA mel spectrogram images. |
| [tga_whisper](examples/tga_whisper) | Transcribe precomputed TGA mel spectrograms with whisper.cpp. |
| [stream_whisper](examples/stream_whisper) | Stream ffmpeg audio through mel, VAD, and Whisper. |
| [vad_ten_eval](examples/vad_ten_eval) | Evaluate `mel-spec` VAD against the vendored TEN-VAD testset. |

## Voice Activity Detection

`mel-spec` includes a lightweight, model-free VAD. It does not load a neural VAD
runtime; it looks for speech-like Sobel edge structure in mel spectrogram frames
and can attach STFT-derived timestamps to each decision.

Current balanced default on the checked-in TEN-VAD testset:

| System | Setting | Macro precision | Macro recall | Macro F1 | Macro FPR | RTFx |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mel-spec` | balanced default | 0.8751 | 0.8785 | 0.8566 | 0.3946 | 819.6 |
| `mel-spec` | high-F1 sweep result | 0.8165 | 0.9635 | 0.8769 | 0.6459 | 828.9 |
| Silero | tuned threshold `0.13` | 0.8897 | 0.9388 | 0.9088 | 0.3602 | 110.3 |
| Silero | default threshold `0.50` | 0.9379 | 0.8630 | 0.8826 | 0.1778 | 110.6 |

The balanced default is not trying to beat learned VADs at strict endpointing.
It is a fast built-in option that reuses ASR mel features and avoids another
model dependency. Tuned Silero is still more accurate overall; TEN-VAD is the
source of the labels and upstream reports stronger precision/recall than Silero
and WebRTC on the same testset.

Detailed method, provenance, commands, speed notes, and per-file results are in
[doc/vad/README.md](doc/vad/README.md).

## TGA Spectrograms

TGA spectrograms are useful when you want a simple interchange format for mel
features. They can be inspected as images, spliced, stored, and passed to the
Whisper examples without keeping the original audio around.

This path is now live in Hush as local browser ASR. The browser uses
`mel-spec`'s Whisper-compatible log-mel output, stores captured speech segments
as compact 8-bit TGA images, decodes them back to an 80-mel `Float32Array`, and
passes that tensor directly to a custom `whisper.cpp` WASM binding via
`whisper_set_mel`. The active Hush deployment verifies that local WASM Whisper
can transcribe from the mel tensor without posting microphone audio to a server.

![image](doc/cutsec_46997.png)
_"the quest for peace."_

Mel spectrograms are also robust under heavy quantization. Whisper does not need
high-precision PCM once the signal has been projected into mel space: 8-bit TGA
images preserve the information the model sees, and even coarse rounding of mel
values can retain useful transcription quality.

```text
Original: [0.158, 0.266, 0.076, 0.196, 0.167, ...]
Rounded:  [0.2,   0.3,   0.1,   0.2,   0.2,   ...]
```

![original quantized mel spectrogram](doc/quantized_mel.png)
![coarsely rounded quantized mel spectrogram](doc/quantized_mel_e1.png)
_(top: original mel values, bottom: values rounded to 1.0e-1 before image
quantization)_

## Performance

Benchmarks on Apple M1 Pro, single-threaded release build:

| Audio Length | Frames | Time | Throughput |
| --- | ---: | ---: | ---: |
| 10s | 997 | 21ms | 476x realtime |
| 60s | 5997 | 124ms | 484x realtime |
| 300s | 29997 | 622ms | 482x realtime |

The CPU path is the default and is already fast enough for many streaming and
batch workloads. Experimental native GPU backends are available behind feature
flags:

| Feature | Backend |
| --- | --- |
| `cuda` | NVIDIA-only cuFFT plus CUDA mel projection. |
| `wgpu` | Native Rust GPU backend for Metal, Vulkan, and DX12 systems. |

## Build Checks

The top-level library tests do not automatically build every standalone example
crate. Use these commands when changing examples:

```bash
cargo test --release
cargo build --release --manifest-path examples/mel_tga/Cargo.toml
cargo build --release --manifest-path examples/stream_whisper/Cargo.toml
cargo build --release --manifest-path examples/tga_whisper/Cargo.toml
cargo run --release --manifest-path examples/vad_ten_eval/Cargo.toml
(cd examples/browser && npm ci && npm test)
```

The Whisper examples compile against the `wavey-ai/whisper-rs` fork and require
a GGML Whisper model to run inference.

## Hush Demo

The Hush live browser demo is active at:

```text
https://wavey.ai/code/hush/?v=20260515-35
```

The source remains at [wavey-ai/hush](https://github.com/wavey-ai/hush). With
the current tuned settings it works as a live browser VAD, spectrogram debugging
view, and local Whisper WASM transcription demo. It exposes mel structure, Sobel
edges, ridge tracks, candidate speech regions, and the local transcript in real
time. The VAD itself should still be treated as experimental rather than a
drop-in replacement for a learned VAD; one strong use case is as a browser-side
feature/debugging front end or cheap prefilter before a stronger VAD/ASR model.

![image](doc/browser.png)
