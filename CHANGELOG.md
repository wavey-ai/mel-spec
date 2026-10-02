# Version 0.5.0

* Bumped the crate as a minor release because feature values change by FFT
  rounding. The public API is compatible with 0.4.1.
* The README release note gives the speedups.

FFT:

* Changed all CPU STFT paths to a real-input FFT from the `realfft` crate.
  The real-input FFT does approximately half the work of a complex FFT.
  `Spectrogram`, `BatchLogMelSpectrogram`, and `Fbank` use it.
* Added `realfft` 3.5 as a dependency.
* `Spectrogram::add` and `Spectrogram::compute_all_cpu` still return all
  `fft_size` bins. The upper bins are the complex conjugates of the lower bins.

Mel projection:

* `SparseMelFilterbank` keeps the weights of each mel row as one contiguous
  band, in `f64` and `f32` forms. The projection adds the band in bin order,
  as before.
* `BatchLogMelSpectrogram`, `Spectrogram::compute_mel_spectrogram_cpu`, and
  `Fbank::compute` project eight frames at a time. Each frame uses one SIMD lane
  and keeps its own summation order.
* `MelSpectrogram::add` calculates the power of each FFT bin one time per frame.
  Before, it calculated the power again for each filter that used the bin.

Batch features:

* `BatchLogMelSpectrogram` writes the pre-emphasized waveform directly into the
  padded buffer. This removes one copy of the input.
* `BatchLogMelSpectrogram` adds the per-feature sums during the frame loop. The
  variance pass processes eight mel rows together and keeps the frame order of
  each sum.
* Fixed a panic when a `BatchLogMelScratch` from a frontend with other settings
  is used. The frontend now resizes the scratch buffers.
* Added a `Default` implementation for `BatchLogMelScratch`.
* `Spectrogram::compute_all_cpu` and `Spectrogram::compute_mel_spectrogram_cpu`
  keep one windowed frame in memory at a time.
* `Fbank::compute` calculates the CMN means row by row in frame order. Each
  column sum has the same order as before.

Streaming:

* `Spectrogram::add` makes one allocation per frame, for the returned spectrum.
* `RingBuffer::maybe_mel` keeps the capacity of its sample buffer and uses the
  half spectrum directly.

Voice activity detection:

* `VoiceActivityDetector` keeps the last `min_x` frames and reuses their
  storage. Before, it kept up to 128 frames.
* `VoiceActivityDetector` reuses its classification buffers across calls.
* For single-column frames, `VoiceActivityDetector` keeps the classification of
  each column triple. A new frame then needs one new classification. Other
  frame shapes use the full window classification.
* Fixed a panic in `vad_boundaries` and `VoiceActivityDetector` for frames in
  column-major layout. For example, `ndarray::concatenate` along axis 1 can
  return a column-major array. The 0.4.0 and 0.4.1 releases have this panic.

Other changes:

* `interleave_frames` reads each frame one time and writes into a zeroed output.
  Before, it copied all frames two times.
* The WASM `SpeechToMel` keeps its two sparse filterbanks. Before, it built
  both filterbanks from dense matrices for each frame.
* The WASM `SpeechToMel` calculates the VAD mel frame only when VAD is on.
* `quant::tga_8bit_data` removes one copy of the input.
  `quant::quantize` allocates its output one time.

Results:

* The real-input FFT changes the FFT rounding. The other changes keep the
  previous arithmetic and summation order.
* On the JFK sample, the `f64` paths differ from `0.4.1` by a maximum of
  8.4e-14.
* On the JFK sample, the Parakeet configuration of `BatchLogMelSpectrogram`
  differs from `0.4.1` by a median of 1.3e-7. The 99th percentile is 1.7e-5,
  and the maximum is 4.5e-4. Low-energy bins have the largest differences.
* `Fbank::compute`, `Spectrogram::compute_mel_spectrogram_cpu`, VAD decisions,
  and `interleave_frames` give the same `f32` output as `0.4.1` on the JFK
  sample.
* The TEN-VAD evaluation gives the same results for each file as `0.4.1`.

Tests:

* Added a test that compares `BatchLogMelSpectrogram` with a frame-by-frame
  reference for seven configurations and ten input lengths.
* Added a test that compares `Fbank::compute` with a frame-by-frame dense
  filterbank reference.
* Added a test that compares `Spectrogram::compute_mel_spectrogram_cpu` with
  `MelSpectrogram::add`.
* Added a test that compares streaming VAD with `vad_boundaries` for
  `min_x` values 0, 1, 2, 3, 5, and 10, with single-column and two-column frames.
* Added a test that compares the real-input FFT spectra with a complex FFT for
  FFT sizes 400, 512, and 9.
* Added a test for the padding layout of `interleave_frames`.
* Added seeded random differential tests. Each test compares an optimized
  path with an independent reference implementation. The tests cover the
  sparse projection, `BatchLogMelSpectrogram`, the Whisper mel paths,
  `Spectrogram`, `Fbank`, `RingBuffer`, the VAD, and `interleave_frames`.
* Set `MEL_SPEC_FUZZ_SCALE` to multiply the number of random cases, for example
  `MEL_SPEC_FUZZ_SCALE=50 cargo test --release fuzz_`.
* Added regression fixtures from 0.4.1 for the JFK sample in
  `testdata/regression`. The streaming VAD decisions must match the fixtures
  exactly.
* `Fbank::compute` must stay within 1e-4 of the fixture, and
  `Spectrogram::compute_mel_spectrogram_cpu` must stay within 1e-5. On the
  platform that made the fixtures, both match exactly. The tolerances allow for
  math-library differences on other platforms.
* The Parakeet configuration of `BatchLogMelSpectrogram` must stay within a
  maximum difference of 5e-3 and a mean difference of 1e-5.

# Version 0.4.1

* Added direct batch feature extraction from interleaved multichannel PCM.
* Fixed oversized ring-buffer writes to retain the newest samples.
* Made the `rtrb` backend use the same overflow behavior as the default backend.
* Enabled `wasm32` builds without browser bindings.
* Limited CUDA compilation to Linux and Windows targets.
* Added the standard CUDA library path for Windows installations.
* Removed production dependencies that were only necessary for tests.

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
