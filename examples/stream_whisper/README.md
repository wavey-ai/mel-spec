# Real-time example with whisper.cpp

Pipe in audio from file:

```
ffmpeg -hide_banner -loglevel error -i ../../testdata/JFKWHA-001-AU_WR.mp3 -f f32le -ar 16000 -acodec pcm_f32le -ac 1 pipe:1  | ./target/debug/stream_whisper --energy-threshold=1.0 --intersection-threshold=10 --min-frames=100 --min-intersections=10 --min-mel=0
```

or microphone:

```
ffmpeg -hide_banner -loglevel error -f avfoundation -i ":1" -f f32le -ar 16000 -acodec pcm_f32le -ac 1 pipe:1 | ./target/debug/stream_whisper --energy-threshold=1.0 --intersection-threshold=10 --min-frames=100 --min-intersections=10 --min-mel=0
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

This is a WIP but results for JFK inaugural address are promising -

```
mel_spec/examples/whisper on  pipeline [✘!+?] is 📦 v0.1.0 via 🦀 v1.72.0-nightly on ☁️  (us-east-1) took 7s
❯ ffmpeg -hide_banner -loglevel error -f avfoundation -i ":1" -f f32le -ar 16000 -acodec pcm_f32le -ac 1 pipe:1  | ./target/debug/mel_spec_example
whisper_init_from_file_no_state: loading model from './../../../whisper.cpp/models/ggml-medium.en.bin'
whisper_model_load: loading model
whisper_model_load: n_vocab       = 51864
whisper_model_load: n_audio_ctx   = 1500
whisper_model_load: n_audio_state = 1024
whisper_model_load: n_audio_head  = 16
whisper_model_load: n_audio_layer = 24
whisper_model_load: n_text_ctx    = 448
whisper_model_load: n_text_state  = 1024
whisper_model_load: n_text_head   = 16
whisper_model_load: n_text_layer  = 24
whisper_model_load: n_mels        = 80
whisper_model_load: ftype         = 1
whisper_model_load: qntvr         = 0
whisper_model_load: type          = 4
whisper_model_load: mem required  = 1899.00 MB (+   43.00 MB per decoder)
whisper_model_load: adding 1607 extra tokens
whisper_model_load: model ctx     = 1462.58 MB
whisper_model_load: model size    = 1462.12 MB
whisper_init_state: kv self size  =   42.00 MB
whisper_init_state: kv cross size =  140.62 MB
whisper_init_state: loading Core ML model from './../../../whisper.cpp/models/ggml-medium.en-encoder.mlmodelc'
whisper_init_state: first run on a device may take a while ...
whisper_init_state: Core ML model loaded
1136 [00:00:11.360]  once upon a time.
```
