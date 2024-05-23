# Real-time example with whisper.cpp

Pipe in audio from file:

```sh
ffmpeg -hide_banner -loglevel error -i ../../testdata/JFKWHA-001-AU_WR.mp3 -f f32le -ar 16000 -acodec pcm_f32le -ac 1 pipe:1  | ./target/debug/stream_whisper
```

or microphone:

```sh
ffmpeg -hide_banner -loglevel error -f avfoundation -i ":1" -f f32le -ar 16000 -acodec pcm_f32le -ac 1 pipe:1 | ./target/debug/stream_whisper
```

As a temporary measure please use the [whisper-rs
fork](https://github.com/wavey-ai/whisper-rs)

```
whisper-rs = { path = "../../whisper-rs", features = ["coreml"]}
```

## usage

*IMPORTANT* TARGA mel spectrogram images used for inference will saved in
`mel_out` for debugging purposes (you can re-run inference on each one,
frame indexes are printed first in the stdout output. However, this folder
will grow quickly if left running.

Transcribed speech will be printed to stdout.
