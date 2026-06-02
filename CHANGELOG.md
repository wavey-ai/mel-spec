# Version 0.4.0
* Bumped the crate as a minor release. The public dense filterbank APIs remain
  available, and the release adds/optimizes execution paths rather than making a
  breaking API change.
* Moved CPU mel projection onto sparse filterbank execution derived from the same
  dense reference matrices:
  - `MelSpectrogram::add` now precomputes and reuses sparse mel weights.
  - `log_mel_spectrogram` now projects through a sparse view of the supplied
    dense filterbank matrix.
  - `Spectrogram::compute_mel_spectrogram_cpu` now routes through
    `MelSpectrogram`, so the legacy CPU batch helper benefits from the same
    sparse projection path.
* Added `BatchLogMelSpectrogram`, `BatchLogMelConfig`, `BatchLogMelScratch`, and
  `BatchLogMelOutput` under the existing `mel` module for whole-utterance ASR
  frontend use cases. The batch frontend keeps FFT, waveform, power, and mel
  buffers alive across calls and supports centered framing, pre-emphasis, log
  guards, padding, and per-feature normalization without introducing any
  model-named API.
* Optimized Kaldi-style `Fbank::compute`:
  - Dense Kaldi filterbank matrices are still built and retained as the
    reference/interchange representation.
  - Runtime projection now uses sparse weights mechanically derived from the
    dense matrix.
  - Power-spectrum and mel-energy buffers are reused during a compute call.
  - Added `Fbank::dense_filterbank()` for reference/export/debug inspection.
* Added explicit consistency tests:
  - Whisper/librosa dense mel fixture comparison still validates
    `testdata/mel_filters.npz`.
  - NeMo dense mel fixture comparison still validates
    `testdata/nemo_mel_filters.npz`.
  - Sparse mel projection is checked against dense projection for every mel bin.
  - Sparse Kaldi fbank projection is checked against dense projection for every
    mel bin.
  - `MelSpectrogram::add` is checked against the legacy dense
    `log_mel_spectrogram + norm_mel` result.
* Documented the dense-vs-sparse contract: dense matrices remain the source of
  truth for compatibility and fixtures; sparse projection is a derived execution
  form, not a separate filterbank definition.
* Added Parakeet/NeMo frontend benchmark notes. On the JFK sample on the M1 Mac,
  the pure Rust `mel-spec` frontend now benchmarks close to the C/libtorch
  TorchScript CPU trace:
  - `mel-spec`: `128x1101`, mean `2.341 ms`, p50 `2.334 ms`, p95 `2.406 ms`,
    `4699.62x` realtime.
  - TorchScript CPU trace: `128x1101`, mean `2.244 ms`, p50 `2.206 ms`,
    p95 `2.813 ms`, `4902.22x` realtime.
  - Full-tensor comparison remains close: MAE `0.001183`, RMSE `0.023699`, max
    absolute error `3.965733`, correlation `0.999719`.
* Added an experimental in-tree `cuda` backend for batched mel spectrograms on NVIDIA systems using cuFFT and a CUDA mel kernel
* Added an experimental native `wgpu` backend for batched mel spectrograms on GPU-capable systems, including Apple Silicon via Metal
* Added `Spectrogram::compute_all_cpu` and `Spectrogram::compute_mel_spectrogram_cpu` batch helpers for CPU/GPU comparisons
* Added a Bluestein-based non-power-of-two GPU FFT path so Whisper's `fft_size = 400` works on the experimental `wgpu` backend

# Version 0.3.4
* Fixed kaldi fbank parity with kaldi_native_fbank:
  - Povey window (like Hamming but goes to zero at edges)
  - Kaldi mel scale (1127 * ln) instead of HTK (2595 * log10)
  - Proper energy floor using f32::EPSILON
* Added performance benchmarks to README (~480x realtime on M1 Pro)
* Documented GPU acceleration options (NeMo, torchaudio, experimental gpu branch)
* Updated examples to use wavey-ai/whisper-rs fork with set_mel + empty samples support
* Rewrote examples to use current mel_spec API (removed mel_spec_pipeline dependency)
* Fixed tga_whisper and stream_whisper to work with pre-computed mel spectrograms

# Version 0.3.3
* Maintenance release

# Version 0.3.0
* Removed mel_spec_pipeline and mel_spec_audio crates
* Simplified API - use Spectrogram and MelSpectrogram directly

# Version 0.2.2
* Voice Activity and word boundary detection enhancements and tests

# Version 0.2.1
* make api public - woops
* add ffmpeg -> mel_spec -> whisper cli example
* split up into modules

# Version 0.2.0
* split into mel, stft mods
* add 8-bit quantisation to marshal to and from greyscale (.tga)
