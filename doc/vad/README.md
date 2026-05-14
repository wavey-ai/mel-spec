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

Best macro-F1 `mel-spec` run from the local sweep:

```bash
cd examples/vad_ten_eval
cargo run --release -- \
  --n-mels 80 \
  --min-energy 1.0 \
  --min-y 8 \
  --min-x 5 \
  --min-speech-ms 200 \
  --merge-gap-ms 150
```

More balanced `mel-spec` run with lower false positives:

```bash
cargo run --release -- \
  --n-mels 80 \
  --min-energy 1.0 \
  --min-y 10 \
  --min-x 5 \
  --min-speech-ms 100 \
  --merge-gap-ms 100
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
| `mel-spec` best macro F1 | `n_mels=80 min_y=8 min_x=5 min_speech=200ms merge_gap=150ms` | 0.8097 | 0.9681 | 0.8765 | 0.6680 | 0.001143 | 875.0 |
| `mel-spec` lower-FPR candidate | `n_mels=80 min_y=10 min_x=5 min_speech=100ms merge_gap=100ms` | 0.8665 | 0.8597 | 0.8442 | 0.4159 | 0.001140 | 877.5 |
| Silero tuned | threshold `0.13` | 0.8897 | 0.9388 | 0.9088 | 0.3602 | 0.009063 | 110.3 |
| Silero default | threshold `0.50` | 0.9379 | 0.8630 | 0.8826 | 0.1778 | 0.009044 | 110.6 |

The best `mel-spec` macro-F1 setting is about 7.9x faster than Silero on this
host, but it reaches that recall by accepting many more false positives. The
lower-FPR `mel-spec` setting is still much faster than Silero, but gives up
accuracy.

TEN-VAD is the source of the labeled testset. Its upstream README reports that
TEN-VAD has a stronger precision/recall curve than Silero and WebRTC on this
same testset, with published CPU RTF values around `0.0086` to `0.0150` and a
library size of about `306KB`. We did not run the TEN binary locally in this
measurement.

## Per-File Results

The table compares the best macro-F1 `mel-spec` run against tuned Silero.

| File | Dur s | mel F1 | mel P | mel R | mel FPR | mel RTFx | Silero F1 | Silero P | Silero R | Silero FPR | Silero RTFx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| testset-audio-01.wav | 11.520 | 0.9188 | 0.8521 | 0.9968 | 0.7714 | 835.8 | 0.9223 | 0.8769 | 0.9727 | 0.5970 | 51.0 |
| testset-audio-02.wav | 4.045 | 0.7742 | 0.6316 | 1.0000 | 1.0000 | 821.5 | 0.9128 | 0.9855 | 0.8500 | 0.0217 | 104.5 |
| testset-audio-03.wav | 10.333 | 0.9427 | 0.8965 | 0.9940 | 0.4774 | 784.1 | 0.9189 | 0.8557 | 0.9922 | 0.6615 | 116.5 |
| testset-audio-04.wav | 10.333 | 0.8723 | 0.8790 | 0.8657 | 0.6319 | 840.1 | 0.9064 | 0.8949 | 0.9182 | 0.5472 | 116.7 |
| testset-audio-05.wav | 10.333 | 0.9390 | 0.8882 | 0.9960 | 0.3394 | 866.1 | 0.9530 | 0.9137 | 0.9957 | 0.2500 | 116.7 |
| testset-audio-06.wav | 10.333 | 0.8921 | 0.8053 | 1.0000 | 1.0000 | 896.6 | 0.9531 | 0.9606 | 0.9457 | 0.1562 | 116.7 |
| testset-audio-07.wav | 8.440 | 0.8316 | 0.7297 | 0.9665 | 0.7491 | 889.9 | 0.8889 | 0.8037 | 0.9944 | 0.5000 | 116.1 |
| testset-audio-08.wav | 9.600 | 0.9461 | 0.9656 | 0.9275 | 0.1548 | 898.2 | 0.9398 | 0.8963 | 0.9878 | 0.5185 | 116.6 |
| testset-audio-09.wav | 10.333 | 0.8454 | 0.7468 | 0.9741 | 1.0000 | 882.1 | 0.7093 | 1.0000 | 0.5496 | 0.0000 | 116.4 |
| testset-audio-10.wav | 10.333 | 0.8311 | 0.7131 | 0.9958 | 0.8931 | 898.0 | 0.8682 | 0.7897 | 0.9640 | 0.5700 | 116.6 |
| testset-audio-11.wav | 8.832 | 0.9552 | 0.9299 | 0.9818 | 0.3292 | 870.3 | 0.9440 | 0.9125 | 0.9777 | 0.4118 | 116.3 |
| testset-audio-12.wav | 4.790 | 0.8055 | 0.6744 | 1.0000 | 0.7790 | 888.7 | 0.8955 | 0.8182 | 0.9890 | 0.3448 | 116.9 |
| testset-audio-13.wav | 10.333 | 0.9569 | 0.9390 | 0.9754 | 0.1929 | 884.7 | 0.9640 | 0.9377 | 0.9918 | 0.2025 | 116.4 |
| testset-audio-14.wav | 6.805 | 0.9203 | 0.8524 | 1.0000 | 0.6739 | 875.8 | 0.9435 | 0.8978 | 0.9940 | 0.4318 | 117.5 |
| testset-audio-15.wav | 4.736 | 0.8430 | 0.7286 | 1.0000 | 1.0000 | 894.3 | 0.9013 | 0.8268 | 0.9906 | 0.5366 | 117.5 |
| testset-audio-16.wav | 10.240 | 0.9434 | 0.9309 | 0.9562 | 0.3468 | 896.7 | 0.9270 | 0.9671 | 0.8902 | 0.1429 | 116.9 |
| testset-audio-17.wav | 3.880 | 0.8523 | 0.7426 | 1.0000 | 0.9143 | 895.4 | 0.8431 | 0.7350 | 0.9885 | 0.9118 | 117.7 |
| testset-audio-18.wav | 7.296 | 0.9092 | 0.8388 | 0.9926 | 0.5598 | 897.1 | 0.9071 | 0.8469 | 0.9765 | 0.5263 | 116.1 |
| testset-audio-19.wav | 9.240 | 0.8839 | 0.7919 | 1.0000 | 1.0000 | 909.4 | 0.9296 | 0.9008 | 0.9604 | 0.3934 | 116.9 |
| testset-audio-20.wav | 10.333 | 0.9633 | 0.9454 | 0.9819 | 0.2374 | 889.2 | 0.9698 | 0.9449 | 0.9961 | 0.2344 | 117.6 |
| testset-audio-21.wav | 3.430 | 0.7745 | 0.6320 | 1.0000 | 1.0000 | 896.3 | 0.9403 | 0.9403 | 0.9403 | 0.1000 | 117.5 |
| testset-audio-22.wav | 14.080 | 0.8060 | 0.6751 | 1.0000 | 0.9636 | 859.1 | 0.8957 | 0.8134 | 0.9966 | 0.4558 | 116.9 |
| testset-audio-23.wav | 4.992 | 0.8654 | 0.7627 | 1.0000 | 1.0000 | 864.7 | 0.8954 | 0.8843 | 0.9068 | 0.3684 | 115.5 |
| testset-audio-24.wav | 6.440 | 0.9162 | 0.8944 | 0.9391 | 0.2865 | 880.3 | 0.9505 | 0.9057 | 1.0000 | 0.2632 | 116.3 |
| testset-audio-25.wav | 15.785 | 0.9520 | 0.9490 | 0.9550 | 0.2124 | 857.2 | 0.9621 | 0.9335 | 0.9924 | 0.2887 | 117.3 |
| testset-audio-26.wav | 10.333 | 0.8448 | 0.7313 | 1.0000 | 1.0000 | 873.3 | 0.9072 | 0.8958 | 0.9188 | 0.2841 | 117.8 |
| testset-audio-27.wav | 8.704 | 0.8025 | 0.6701 | 1.0000 | 1.0000 | 897.4 | 0.7861 | 0.8293 | 0.7473 | 0.3111 | 117.6 |
| testset-audio-28.wav | 7.168 | 0.8569 | 0.9357 | 0.7903 | 0.1638 | 885.2 | 0.9305 | 0.9390 | 0.9222 | 0.1786 | 115.9 |
| testset-audio-29.wav | 8.960 | 0.9003 | 0.8269 | 0.9881 | 0.6376 | 910.1 | 0.9330 | 0.9058 | 0.9619 | 0.3000 | 116.5 |
| testset-audio-30.wav | 10.333 | 0.7497 | 0.7332 | 0.7668 | 0.7263 | 889.0 | 0.8659 | 0.8795 | 0.8528 | 0.2967 | 116.5 |
