### using with whisper.cpp

Please refer to these PRs, both whisper.cpp and whisper-rs require small
changes:

https://github.com/tazz4843/whisper-rs/pull/75
https://github.com/ggerganov/whisper.cpp/pull/1130

### Mel Spec

A Rust implementation of mel spectrograms aligned to the results from the
whisper.cpp, pytorch and librosa reference implementations and suited to
streaming audio.

The main objective is to allow inference from spectrograms alone, so that
audio samples don't need to be kept in context for follow-up processing.

#### filter banks

Mel filter banks are within 1.0e-7 of `librosa.filters.mel` and identical to
the GGML model-embedded filters used by whisper.cpp.

#### stft

A stft implementation that allows creating spectrograms from an audio steam -
near identical to those produced by whisper.cpp internally.

An example of whisper inference from mel spectrograms via `whisper-rs` can be
found in the tests.
