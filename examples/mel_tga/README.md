# mel_tga - Audio to Mel Spectrogram TGA

Converts audio to mel spectrograms and saves them as TGA image files. These spectrograms are like a photographic negative - they can be saved, spliced, and played back by whisper.cpp for transcription without the original audio.

## Building

From the repository root:

```sh
cargo build --release --manifest-path examples/mel_tga/Cargo.toml
```

## Usage

From the repository root, pipe raw f32le audio (16kHz mono) via stdin:

```sh
cargo build --release --manifest-path examples/mel_tga/Cargo.toml
ffmpeg -i input.wav -f f32le -ar 16000 -ac 1 pipe:1 | ./examples/mel_tga/target/release/mel_tga
```

From this example directory, the same binary is:

```sh
ffmpeg -i audio.mp3 -f f32le -ar 16000 -ac 1 pipe:1 | ./target/release/mel_tga
```

### Options

- `-m, --mels <n>` - Number of mel bins (default: `80`)
- `-o, --out-dir <path>` - Output directory (default: `./mel_out`)

### Output

Will output `.tga` files, chunking them if the width exceeds `u16::MAX` (the max TARGA width supported by their u16 headers):

```
out_chunk0.tga  out_chunk1.tga
```

## Notes

- Input must be raw f32le audio at 16kHz mono sample rate
- The example buffers the entire output and waits for the pipe to close before writing TGA files
- TGA files can be transcribed using `tga_whisper`
