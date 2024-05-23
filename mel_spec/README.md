# Mel Spec

A Rust implementation of mel spectrograms aligned to the results from the
whisper.cpp, pytorch and librosa reference implementations and suited to
streaming audio.

## Usage

To require the libary's main features:

```
use mel_spec::prelude::*
```

### Mel filterbank that has parity with librosa:

Mel filterbanks, within 1.0e-7 of librosa and identical to whisper
GGML model-embedded filters.

```rust
    let file_path = "./testdata/mel_filters.npz";
    let f = File::open(file_path).unwrap();
    let mut npz = NpzReader::new(f).unwrap();
    let filters: Array2<f32> = npz.by_index(0).unwrap();
    let want: Array2<f64> = filters.mapv(|x| f64::from(x));
    let sampling_rate = 16000.0;
    let fft_size = 400;
    let n_mels = 80;
    let f_min = None;
    let f_max = None;
    let hkt = false;
    let norm = true;
    let got = mel(sampling_rate, fft_size, n_mels, f_min, f_max, hkt, norm);
    assert_eq!(got.shape(), vec![80, 201]);
    for i in 0..80 {
        assert_nearby!(got.row(i), want.row(i), 1.0e-7);
    }
```

### Spectrogram using Short Time Fourier Transform

STFT with overlap-and-save that has parity with pytorch and
whisper.cpp.

The implementation is suitable for processing streaming audio and
will accumulate the correct amount of data before returning fft
results.

```rust
    let fft_size = 8;
    let hop_size = 4;
    let mut spectrogram = Spectrogram::new(fft_size, hop_size);

    // Add PCM audio samples
    let frames: Vec<f32> = vec![1.0, 2.0, 3.0];
    if let Some(fft_frame) = spectrogram.add(&frames) {
        // use fft result
    }
```

### STFT Spectrogram to Mel Spectrogram

MelSpectrogram applies a pre-computed filterbank to an FFT result.
Results are identical to whisper.cpp and whisper.py

```rust
    let fft_size = 400;
    let sampling_rate = 16000.0;
    let n_mels = 80;
    let mut mel = MelSpectrogram::new(fft_size, sampling_rate, n_mels);
    // Example input data for the FFT
    let fft_input = Array1::from(vec![Complex::new(1.0, 0.0); fft_size]);
    // Add the FFT data to the MelSpectrogram
    let mel_spec = stage.add(fft_input);
```

### Creating Mel Spectrograms from Audio.

The library includes basic audio helper and a pipeline for processing
PCM audio and creating Mel spectrograms that can be sent to whisper.cpp.

It also has voice activity detection that uses edge detection (which
might be a novel approach) to identify word/speech boundaries in real-
time.

```rust
    // load the whisper jfk sample
    let file_path = "../testdata/jfk_f32le.wav";
    let file = File::open(&file_path).unwrap();
    let data = parse_wav(file).unwrap();
    let samples = deinterleave_vecs_f32(&data.data, 1);

    let fft_size = 400;
    let hop_size = 160;
    let n_mels = 80;
    let sampling_rate = 16000.0;

    let mel_settings = MelConfig::new(fft_size, hop_size, n_mels, sampling_rate);
    let vad_settings = DetectionSettings::new(1.0, 10, 5, 0, 100);

    let config = PipelineConfig::new(mel_settings, Some(vad_settings));

    let mut pl = Pipeline::new(config);

    let handles = pl.start();

    // chunk size can be anything, 88 is random
    for chunk in samples[0].chunks(88) {
        let _ = pl.send_pcm(chunk);
    }

    pl.close_ingress();

    while let Ok((_, mel_spectrogram)) = pl.rx().recv() {
      // do something with spectrogram
    }
```

### Saving Mel Spectrograms to file

Mel spectrograms can be saved in Tga format - an uncompressed image format
supported by OSX and Windows.

As these images directly encode quantized mel spectrogram data they represent
a "photographic negative" of audio data that whisper.cpp can develop and print
without the need for direct audio input.

`tga` files are used in lieu of actual audio for most of the library tests. These
files are lossless in Speech-to-Text terms, they encode all the information that
is available in the model's view of raw audio and will produce identical results.

Note that spectrograms must have an even number of columns in the time domain,
otherwise Whisper will hallucinate. the library takes care of this if using the
core methods.

```
     let file_path = "../testdata/jfk_full_speech_chunk0_golden.tga";
     let dequantized_mel = load_tga_8bit(file_path).unwrap();
     // dequantized_mel can be sent straight to whisper.cpp
```

```
❯ ffmpeg -hide_banner -loglevel error -i ~/Downloads/JFKWHA-001-AU_WR.mp3 -f f32le -ar 16000 -acodec pcm_f32le -ac 1 pipe:1  | ./target/debug/tga_whisper -t ../../doc/cutsec_46997.tga
...
whisper_init_state: Core ML model loaded
Got 1
 the quest for peace.
```

![image](doc/cutsec_46997.png)
_the quest for peace._

### Voice Activity Detection

I had the idea of using the Sobel operator for this as speech in Mel spectrograms
is characterised by clear gradients.

The general idea is to outline structure in the spectrogram and then find vertical
gaps that are suitable for cutting - to allow passing new spectrograms to the model
in near real-time.

See the README in the main repo for more details.
