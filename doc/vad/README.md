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

Balanced `mel-spec` VAD preset:

```bash
cd examples/vad_ten_eval
cargo run --release
```

The explicit default values are `n_mels=80`, `min_energy=0.98`, `min_y=11`,
`min_x=5`, `min_mel=2`, `min_speech_ms=150`, and `merge_gap_ms=150`.

High-F1 sweep result with more false positives:

```bash
cargo run --release -- \
  --n-mels 80 \
  --min-energy 0.96 \
  --min-y 8 \
  --min-x 5 \
  --min-mel 4 \
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
| `mel-spec` balanced default | `n_mels=80 min_energy=0.98 min_y=11 min_x=5 min_mel=2 min_speech=150ms merge_gap=150ms` | 0.8751 | 0.8785 | 0.8566 | 0.3946 | 0.001220 | 819.6 |
| `mel-spec` high-F1 sweep result | `n_mels=80 min_energy=0.96 min_y=8 min_x=5 min_mel=4 min_speech=200ms merge_gap=150ms` | 0.8165 | 0.9635 | 0.8769 | 0.6459 | 0.001206 | 828.9 |
| Silero tuned | threshold `0.13` | 0.8897 | 0.9388 | 0.9088 | 0.3602 | 0.009063 | 110.3 |
| Silero default | threshold `0.50` | 0.9379 | 0.8630 | 0.8826 | 0.1778 | 0.009044 | 110.6 |

The balanced default is the better VAD default from this sweep: it improves the
previous lower-FPR preset on macro F1 and false positives while remaining about
7.4x faster than Silero on this host. The high-F1 sweep result is useful when
missed speech is more expensive than sending extra non-speech audio, but it
accepts many more false positives.

TEN-VAD is the source of the labeled testset. Its upstream README reports that
TEN-VAD has a stronger precision/recall curve than Silero and WebRTC on this
same testset, with published CPU RTF values around `0.0086` to `0.0150` and a
library size of about `306KB`. We did not run the TEN binary locally in this
measurement.

## Per-File Results

The table compares the balanced `mel-spec` default against tuned Silero.

| File | Dur s | mel F1 | mel P | mel R | mel FPR | mel RTFx | Silero F1 | Silero P | Silero R | Silero FPR | Silero RTFx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| testset-audio-01.wav | 11.520 | 0.9479 | 0.9080 | 0.9915 | 0.4476 | 818.6 | 0.9223 | 0.8769 | 0.9727 | 0.5970 | 51.0 |
| testset-audio-02.wav | 4.045 | 0.8386 | 0.7221 | 1.0000 | 0.6599 | 860.8 | 0.9128 | 0.9855 | 0.8500 | 0.0217 | 104.5 |
| testset-audio-03.wav | 10.333 | 0.9398 | 1.0000 | 0.8865 | 0.0000 | 816.4 | 0.9189 | 0.8557 | 0.9922 | 0.6615 | 116.5 |
| testset-audio-04.wav | 10.333 | 0.7408 | 0.9460 | 0.6088 | 0.1840 | 847.2 | 0.9064 | 0.8949 | 0.9182 | 0.5472 | 116.7 |
| testset-audio-05.wav | 10.333 | 0.9637 | 0.9915 | 0.9373 | 0.0217 | 788.6 | 0.9530 | 0.9137 | 0.9957 | 0.2500 | 116.7 |
| testset-audio-06.wav | 10.333 | 0.8921 | 0.8053 | 1.0000 | 1.0000 | 850.6 | 0.9531 | 0.9606 | 0.9457 | 0.1562 | 116.7 |
| testset-audio-07.wav | 8.440 | 0.8259 | 0.8547 | 0.7989 | 0.2841 | 829.6 | 0.8889 | 0.8037 | 0.9944 | 0.5000 | 116.1 |
| testset-audio-08.wav | 9.600 | 0.8458 | 0.9815 | 0.7430 | 0.0655 | 835.7 | 0.9398 | 0.8963 | 0.9878 | 0.5185 | 116.6 |
| testset-audio-09.wav | 10.333 | 0.7209 | 0.7421 | 0.7008 | 0.7373 | 868.4 | 0.7093 | 1.0000 | 0.5496 | 0.0000 | 116.4 |
| testset-audio-10.wav | 10.333 | 0.8829 | 0.7939 | 0.9944 | 0.5755 | 830.6 | 0.8682 | 0.7897 | 0.9640 | 0.5700 | 116.6 |
| testset-audio-11.wav | 8.832 | 0.9035 | 1.0000 | 0.8240 | 0.0000 | 838.5 | 0.9440 | 0.9125 | 0.9777 | 0.4118 | 116.3 |
| testset-audio-12.wav | 4.790 | 0.9171 | 0.8731 | 0.9658 | 0.2265 | 873.7 | 0.8955 | 0.8182 | 0.9890 | 0.3448 | 116.9 |
| testset-audio-13.wav | 10.333 | 0.9533 | 1.0000 | 0.9107 | 0.0000 | 738.7 | 0.9640 | 0.9377 | 0.9918 | 0.2025 | 116.4 |
| testset-audio-14.wav | 6.805 | 0.9492 | 0.9414 | 0.9572 | 0.2319 | 834.2 | 0.9435 | 0.8978 | 0.9940 | 0.4318 | 117.5 |
| testset-audio-15.wav | 4.736 | 0.8834 | 0.7912 | 1.0000 | 0.7087 | 870.8 | 0.9013 | 0.8268 | 0.9906 | 0.5366 | 117.5 |
| testset-audio-16.wav | 10.240 | 0.9590 | 0.9630 | 0.9550 | 0.1792 | 829.4 | 0.9270 | 0.9671 | 0.8902 | 0.1429 | 116.9 |
| testset-audio-17.wav | 3.880 | 0.9106 | 0.8544 | 0.9747 | 0.4381 | 891.9 | 0.8431 | 0.7350 | 0.9885 | 0.9118 | 117.7 |
| testset-audio-18.wav | 7.296 | 0.9715 | 0.9653 | 0.9778 | 0.1033 | 823.7 | 0.9071 | 0.8469 | 0.9765 | 0.5263 | 116.1 |
| testset-audio-19.wav | 9.240 | 0.8839 | 0.7919 | 1.0000 | 1.0000 | 780.3 | 0.9296 | 0.9008 | 0.9604 | 0.3934 | 116.9 |
| testset-audio-20.wav | 10.333 | 0.9597 | 0.9579 | 0.9614 | 0.1768 | 852.2 | 0.9698 | 0.9449 | 0.9961 | 0.2344 | 117.6 |
| testset-audio-21.wav | 3.430 | 0.8038 | 0.6719 | 1.0000 | 0.8387 | 697.3 | 0.9403 | 0.9403 | 0.9403 | 0.1000 | 117.5 |
| testset-audio-22.wav | 14.080 | 0.8618 | 0.8075 | 0.9241 | 0.4411 | 815.5 | 0.8957 | 0.8134 | 0.9966 | 0.4558 | 116.9 |
| testset-audio-23.wav | 4.992 | 0.8694 | 0.7689 | 1.0000 | 0.9658 | 878.8 | 0.8954 | 0.8843 | 0.9068 | 0.3684 | 115.5 |
| testset-audio-24.wav | 6.440 | 0.8176 | 0.9701 | 0.7065 | 0.0562 | 792.1 | 0.9505 | 0.9057 | 1.0000 | 0.2632 | 116.3 |
| testset-audio-25.wav | 15.785 | 0.9281 | 1.0000 | 0.8658 | 0.0000 | 820.9 | 0.9621 | 0.9335 | 0.9924 | 0.2887 | 117.3 |
| testset-audio-26.wav | 10.333 | 0.8622 | 0.7578 | 1.0000 | 0.8696 | 809.4 | 0.9072 | 0.8958 | 0.9188 | 0.2841 | 117.8 |
| testset-audio-27.wav | 8.704 | 0.8193 | 0.6965 | 0.9948 | 0.8807 | 773.6 | 0.7861 | 0.8293 | 0.7473 | 0.3111 | 117.6 |
| testset-audio-28.wav | 7.168 | 0.2634 | 1.0000 | 0.1517 | 0.0000 | 843.9 | 0.9305 | 0.9390 | 0.9222 | 0.1786 | 115.9 |
| testset-audio-29.wav | 8.960 | 0.9157 | 0.8742 | 0.9613 | 0.4266 | 792.9 | 0.9330 | 0.9058 | 0.9619 | 0.3000 | 116.5 |
| testset-audio-30.wav | 10.333 | 0.6683 | 0.8212 | 0.5633 | 0.3193 | 793.3 | 0.8659 | 0.8795 | 0.8528 | 0.2967 | 116.5 |
