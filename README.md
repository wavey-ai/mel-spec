### Mel Spec

Mel filterbank in Rust using `ndarray::Array2` that is within 1.0e-7 of the values produced by
`librosa.filters.mel`.

#### usage

`let filterbank = mel(16000.0, 400, 80)`

This is identical to the mel filters used for Whisper training and inference (`mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)`).

Tested against https://github.com/openai/whisper/whisper/assets/mel_filters.npz
# mel_spec
