# tga_whisper - Transcribe Mel Spectrogram TGA Files

Transcribe audio that has been encoded as a mel spectrogram TGA image. No audio required - the spectrogram contains all the information Whisper needs.

## Building

```sh
cargo build --release
```

## Usage

```sh
./target/release/tga_whisper -m /path/to/ggml-base.en.bin -t spectrogram.tga
```

### Options

- `-m, --model-path <path>` - Path to Whisper GGML model (default: `./../../../whisper.cpp/models/ggml-medium.en.bin`)
- `-t, --tga-path <path>` - Path to TGA mel spectrogram file (required)

### Example

First generate a TGA spectrogram using `mel_tga`:

```sh
ffmpeg -i audio.wav -f f32le -ar 16000 -ac 1 pipe:1 | ../mel_tga/target/release/mel_tga
```

Then transcribe it:

```sh
./target/release/tga_whisper -m ~/ggml-base.en.bin -t ./mel_out/out_chunk0.tga
```

Output:
```
Got 1 segments
 And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

## How It Works

This uses `set_mel()` to load a pre-computed mel spectrogram directly into Whisper, bypassing audio processing entirely. This is enabled by [whisper.cpp PR #1214](https://github.com/ggml-org/whisper.cpp/pull/1214) and our [wavey-ai/whisper-rs](https://github.com/wavey-ai/whisper-rs) fork.

## Download Models

```sh
# Base English model (~150MB)
curl -L -o ~/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# Medium English model (~1.5GB, more accurate)
curl -L -o ~/ggml-medium.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin
```
