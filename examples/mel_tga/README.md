# mel_tga - Audio to Mel Spectrogram TGA

Converts audio to mel spectrograms and saves them as TGA image files. These spectrograms are like a photographic negative - they can be saved, spliced, and played back by whisper.cpp for transcription without the original audio.

## Building

```sh
cargo build --release
```

## Usage

Pipe raw f32le audio (16kHz mono) via stdin:

```sh
ffmpeg -i input.wav -f f32le -ar 16000 -ac 1 pipe:1 | ./target/release/mel_tga
```

Or from an MP3:

```sh
ffmpeg -i audio.mp3 -f f32le -ar 16000 -ac 1 pipe:1 | ./target/release/mel_tga
```

### Options

- `-o, --out-path <path>` - Output directory (default: `./mel_out`)

### Output

Will output `.tga` files, chunking them if the width exceeds `u16::MAX` (the max TARGA width supported by their u16 headers):

```
out_chunk0.tga  out_chunk1.tga
```

## Notes

- Input must be raw f32le audio at 16kHz mono sample rate
- The example buffers the entire output and waits for the pipe to close before writing TGA files
- TGA files can be transcribed using `tga_whisper`
