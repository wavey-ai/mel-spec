# stream_whisper - Real-time Streaming Transcription

Real-time speech-to-text using mel spectrograms with Voice Activity Detection (VAD). Processes audio in chunks and transcribes when speech boundaries are detected.

## Building

```sh
cargo build --release
```

## Usage

### From audio file

```sh
ffmpeg -i audio.mp3 -f f32le -ar 16000 -ac 1 pipe:1 | ./target/release/stream_whisper -m ~/ggml-base.en.bin
```

### From microphone (macOS)

```sh
ffmpeg -f avfoundation -i ":1" -f f32le -ar 16000 -ac 1 pipe:1 | ./target/release/stream_whisper -m ~/ggml-base.en.bin
```

### Options

- `-m, --model-path <path>` - Path to Whisper GGML model
- `-o, --out-path <path>` - Output directory for debug TGA files (default: `./mel_out`)
- `--min-power <float>` - Minimum energy threshold for VAD (default: 1.0)
- `--min-y <int>` - Minimum frequency bins for speech detection (default: 3)
- `--min-x <int>` - Minimum time frames for speech detection (default: 5)
- `--min-mel <int>` - Minimum mel bin index (default: 0)
- `--min-frames <int>` - Minimum frames before processing (default: 100)

## How It Works

1. Audio is processed through STFT -> Mel filterbank
2. Voice Activity Detection finds speech boundaries using Sobel edge detection
3. When a speech boundary is detected, the accumulated mel spectrogram is sent to Whisper
4. Transcription is printed with frame index and timestamp

## Debug Output

TGA mel spectrogram images are saved in the output directory for debugging. Each file corresponds to a transcribed segment.

**Note:** The output directory will grow if left running. Clean it periodically.

## Download Models

```sh
curl -L -o ~/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```
