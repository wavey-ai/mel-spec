# TEN-VAD Labeled VAD Testset

This directory vendors the labeled VAD testset from TEN-VAD so `mel-spec`
VAD evaluation is reproducible without a separate checkout.

## Provenance

- Source repository: <https://github.com/TEN-framework/ten-vad>
- Source commit: `22a3bcd4509d0faaa8eef4881e8af5f39c178950`
- Source path: `testset/`
- Copied into this repository on: 2026-05-14
- Source description: TEN-VAD says these are manually annotated VAD test
  files drawn from sources including LibriSpeech, GigaSpeech, and the DNS
  Challenge.

## Contents

- `testset-audio-01.wav` through `testset-audio-30.wav`
- Matching `testset-audio-XX.scv` label files
- `LICENSE.TEN-VAD`
- `NOTICES.TEN-VAD`

The WAV files are 16 kHz mono PCM audio. Each `.scv` file contains one row:

```text
name,start_seconds,end_seconds,label,...
```

where `label` is `1` for speech and `0` for non-speech.

## License

The upstream TEN-VAD repository is distributed under the license text copied in
`LICENSE.TEN-VAD`, with notices copied in `NOTICES.TEN-VAD`. Keep those files
with this testset if the audio or labels are redistributed.
