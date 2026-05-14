# VAD Evaluation

This document describes the `mel-spec` VAD approach and the current evaluation
against the checked-in TEN-VAD labeled testset.

## Approach

`mel-spec` VAD is not a learned speech/non-speech model. It is a lightweight
deterministic detector over STFT-derived log-mel frames:

1. Compute STFT frames from 16 kHz mono PCM.
2. Project each frame to log-mel features.
3. Run Sobel-style edge detection across the mel/time image.
4. Treat columns with enough speech-like gradient structure as active.
5. Use inactive columns as likely cut points for streaming/chunking.

This makes it useful as a cheap chunk-boundary signal before ASR. It is less
precise than a learned VAD when non-speech regions contain breath, room noise,
or other speech-like spectral structure.

The streaming API can also return timestamps. Given `fft_size`, `hop_size`, and
`sampling_rate`, each VAD decision carries start, center, and end timestamps for
the corresponding STFT frame.

## Testset

The 30-file TEN-VAD testset is vendored at `testdata/ten-vad`.

- Source repository: <https://github.com/TEN-framework/ten-vad>
- Source commit: `22a3bcd4509d0faaa8eef4881e8af5f39c178950`
- Copied path: `testset/`
- Copied date: 2026-05-14
- Total audio measured here: 262.316 seconds

See `testdata/ten-vad/README.md`, `LICENSE.TEN-VAD`, and `NOTICES.TEN-VAD`
for provenance and license details.

## Commands

Recommended `mel-spec` preset for ASR chunking:

```bash
cd examples/vad_ten_eval
cargo run --release
```

The explicit default values are `n_mels=80`, `min_energy=1.0`, `min_y=10`,
`min_x=5`, `min_speech_ms=100`, and `merge_gap_ms=100`.

High-recall sweep result:

```bash
cargo run --release -- \
  --n-mels 80 \
  --min-energy 1.0 \
  --min-y 8 \
  --min-x 5 \
  --min-speech-ms 200 \
  --merge-gap-ms 150
```

Silero was measured with the JIT model from `snakers4/silero-vad` at commit
`bbf22a00640614309d60aba5467189b48c7c6ecc`, using 512-sample frames and the
same label conversion as TEN-VAD's `plot_pr_curves.py`.

## Summary

Measured locally on macOS. `mel-spec` numbers include STFT, mel projection, VAD,
and postprocessing. Silero numbers exclude model load; model load was about
0.066 seconds in this run.

| System | Threshold/config | Macro precision | Macro recall | Macro F1 | Macro FPR | RTF | RTFx |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mel-spec` recommended preset | `n_mels=80 min_y=10 min_x=5 min_speech=100ms merge_gap=100ms` | 0.8665 | 0.8597 | 0.8442 | 0.4159 | 0.001149 | 870.2 |
| `mel-spec` high-recall sweep result | `n_mels=80 min_y=8 min_x=5 min_speech=200ms merge_gap=150ms` | 0.8097 | 0.9681 | 0.8765 | 0.6680 | 0.001143 | 875.0 |
| Silero tuned | threshold `0.13` | 0.8897 | 0.9388 | 0.9088 | 0.3602 | 0.009063 | 110.3 |
| Silero default | threshold `0.50` | 0.9379 | 0.8630 | 0.8826 | 0.1778 | 0.009044 | 110.6 |

The recommended preset is the better operational default for ASR chunking: it
has lower false positives across the 30 TEN files while remaining about 7.9x
faster than Silero on this host. The high-recall sweep result is useful when
missed speech is more expensive than sending extra non-speech audio, but it
accepts many more false positives.

TEN-VAD is the source of the labeled testset. Its upstream README reports that
TEN-VAD has a stronger precision/recall curve than Silero and WebRTC on this
same testset, with published CPU RTF values around `0.0086` to `0.0150` and a
library size of about `306KB`. We did not run the TEN binary locally in this
measurement.

## Per-File Results

The table compares the recommended `mel-spec` preset against tuned Silero.

| File | Dur s | mel F1 | mel P | mel R | mel FPR | mel RTFx | Silero F1 | Silero P | Silero R | Silero FPR | Silero RTFx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| testset-audio-01.wav | 11.520 | 0.9499 | 0.9277 | 0.9733 | 0.3381 | 663.8 | 0.9223 | 0.8769 | 0.9727 | 0.5970 | 51.0 |
| testset-audio-02.wav | 4.045 | 0.8155 | 0.6885 | 1.0000 | 0.7755 | 689.0 | 0.9128 | 0.9855 | 0.8500 | 0.0217 | 104.5 |
| testset-audio-03.wav | 10.333 | 0.9206 | 0.9631 | 0.8816 | 0.1407 | 681.7 | 0.9189 | 0.8557 | 0.9922 | 0.6615 | 116.5 |
| testset-audio-04.wav | 10.333 | 0.6991 | 0.9359 | 0.5579 | 0.2025 | 782.5 | 0.9064 | 0.8949 | 0.9182 | 0.5472 | 116.7 |
| testset-audio-05.wav | 10.333 | 0.9658 | 0.9916 | 0.9413 | 0.0217 | 884.6 | 0.9530 | 0.9137 | 0.9957 | 0.2500 | 116.7 |
| testset-audio-06.wav | 10.333 | 0.8921 | 0.8053 | 1.0000 | 1.0000 | 921.5 | 0.9531 | 0.9606 | 0.9457 | 0.1562 | 116.7 |
| testset-audio-07.wav | 8.440 | 0.8713 | 0.8768 | 0.8660 | 0.2546 | 903.1 | 0.8889 | 0.8037 | 0.9944 | 0.5000 | 116.1 |
| testset-audio-08.wav | 9.600 | 0.8339 | 0.9811 | 0.7252 | 0.0655 | 914.8 | 0.9398 | 0.8963 | 0.9878 | 0.5185 | 116.6 |
| testset-audio-09.wav | 10.333 | 0.7037 | 0.7275 | 0.6813 | 0.7725 | 921.4 | 0.7093 | 1.0000 | 0.5496 | 0.0000 | 116.4 |
| testset-audio-10.wav | 10.333 | 0.8835 | 0.8089 | 0.9732 | 0.5126 | 934.2 | 0.8682 | 0.7897 | 0.9640 | 0.5700 | 116.6 |
| testset-audio-11.wav | 8.832 | 0.8700 | 0.9982 | 0.7709 | 0.0062 | 938.9 | 0.9440 | 0.9125 | 0.9777 | 0.4118 | 116.3 |
| testset-audio-12.wav | 4.790 | 0.8176 | 0.8067 | 0.8288 | 0.3204 | 926.8 | 0.8955 | 0.8182 | 0.9890 | 0.3448 | 116.9 |
| testset-audio-13.wav | 10.333 | 0.9090 | 1.0000 | 0.8331 | 0.0000 | 881.5 | 0.9640 | 0.9377 | 0.9918 | 0.2025 | 116.4 |
| testset-audio-14.wav | 6.805 | 0.9106 | 0.9015 | 0.9199 | 0.3913 | 876.6 | 0.9435 | 0.8978 | 0.9940 | 0.4318 | 117.5 |
| testset-audio-15.wav | 4.736 | 0.8789 | 0.7839 | 1.0000 | 0.7402 | 884.2 | 0.9013 | 0.8268 | 0.9906 | 0.5366 | 117.5 |
| testset-audio-16.wav | 10.240 | 0.9380 | 0.9448 | 0.9314 | 0.2659 | 939.3 | 0.9270 | 0.9671 | 0.8902 | 0.1429 | 116.9 |
| testset-audio-17.wav | 3.880 | 0.8734 | 0.7935 | 0.9711 | 0.6667 | 895.7 | 0.8431 | 0.7350 | 0.9885 | 0.9118 | 117.7 |
| testset-audio-18.wav | 7.296 | 0.9646 | 0.9718 | 0.9574 | 0.0815 | 873.6 | 0.9071 | 0.8469 | 0.9765 | 0.5263 | 116.1 |
| testset-audio-19.wav | 9.240 | 0.8898 | 0.8015 | 1.0000 | 0.9424 | 917.5 | 0.9296 | 0.9008 | 0.9604 | 0.3934 | 116.9 |
| testset-audio-20.wav | 10.333 | 0.9534 | 0.9574 | 0.9493 | 0.1768 | 898.3 | 0.9698 | 0.9449 | 0.9961 | 0.2344 | 117.6 |
| testset-audio-21.wav | 3.430 | 0.8386 | 0.7220 | 1.0000 | 0.6613 | 895.7 | 0.9403 | 0.9403 | 0.9403 | 0.1000 | 117.5 |
| testset-audio-22.wav | 14.080 | 0.8358 | 0.8056 | 0.8684 | 0.4197 | 879.4 | 0.8957 | 0.8134 | 0.9966 | 0.4558 | 116.9 |
| testset-audio-23.wav | 4.992 | 0.8654 | 0.7627 | 1.0000 | 1.0000 | 895.5 | 0.8954 | 0.8843 | 0.9068 | 0.3684 | 115.5 |
| testset-audio-24.wav | 6.440 | 0.6806 | 0.9494 | 0.5304 | 0.0730 | 924.5 | 0.9505 | 0.9057 | 1.0000 | 0.2632 | 116.3 |
| testset-audio-25.wav | 15.785 | 0.9196 | 0.9909 | 0.8579 | 0.0327 | 878.7 | 0.9621 | 0.9335 | 0.9924 | 0.2887 | 117.3 |
| testset-audio-26.wav | 10.333 | 0.8763 | 0.7799 | 1.0000 | 0.7681 | 898.5 | 0.9072 | 0.8958 | 0.9188 | 0.2841 | 117.8 |
| testset-audio-27.wav | 8.704 | 0.8067 | 0.6784 | 0.9948 | 0.9579 | 925.6 | 0.7861 | 0.8293 | 0.7473 | 0.3111 | 117.6 |
| testset-audio-28.wav | 7.168 | 0.4012 | 1.0000 | 0.2509 | 0.0000 | 903.5 | 0.9305 | 0.9390 | 0.9222 | 0.1786 | 115.9 |
| testset-audio-29.wav | 8.960 | 0.9143 | 0.8657 | 0.9688 | 0.4633 | 915.0 | 0.9330 | 0.9058 | 0.9619 | 0.3000 | 116.5 |
| testset-audio-30.wav | 10.333 | 0.6473 | 0.7734 | 0.5566 | 0.4246 | 915.1 | 0.8659 | 0.8795 | 0.8528 | 0.2967 | 116.5 |
