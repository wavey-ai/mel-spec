# Audio to interoperable TARGA Mel Spectrograms

These spectrograms are like a photographic negative as far as Whisper is
concerned, they can be saved, spliced and played back by whisper.cpp.

```
cargo build && ffmpeg -hide_banner -loglevel error -i ../../testdata/JFKWHA-001-AU_WR.mp3 -f f32le -ar 16000 -acodec pcm_f32le -ac 1 pipe:1  | ./target/debug/mel_tga
```

Will output .tga files chunking them if the width exceeds `u16::MAX` which is
the max TARGA width supported by their u16 headers.

```
out_chunk0.tga	out_chunk1.tga
```

As we require the width before writing the header, this example buffers the
entire output and waits for the pipe to close before writing the tga files.
