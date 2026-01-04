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
