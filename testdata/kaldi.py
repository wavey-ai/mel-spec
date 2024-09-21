import numpy as np
import kaldi_native_fbank as knf
import matplotlib.pyplot as plt
from scipy.io import wavfile


def compute_fbank(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.snip_edges = True
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.mel_opts.num_bins = 80
    opts.mel_opts.debug_mel = False

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, samples.tolist())
    fbank.input_finished()

    num_frames = fbank.num_frames_ready
    features = []
    for i in range(num_frames):
        frame = fbank.get_frame(i)
        features.append(frame)
    features = np.stack(features, axis=0)

    # Apply Cepstral Mean Normalization (CMN)
    features = features - np.mean(features, axis=0)

    return features


def main():
    # Load the wave file
    file_path = "../testdata/jfk_f32le.wav"  # Specify the path to your wave file
    sample_rate, samples = wavfile.read(file_path)

    # Ensure samples are in the correct format
    samples = samples.astype(np.float32)
    if len(samples.shape) > 1:
        samples = samples[:, 0]  # Take the first channel if stereo

    # Compute filterbank features
    features = compute_fbank(samples, sample_rate)
    features = features.T
    np.savez("./testdata/kaldi_native_fbank_jfk.npz", features=features)
    print("Python Mel spectrogram shape:", features.shape)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(features, aspect="auto", origin="lower", cmap="viridis")
    plt.title("Mel Spectrogram")
    plt.ylabel("Mel Filter Banks")
    plt.xlabel("Frame (Time)")
    plt.colorbar(label="Magnitude (dB)")
    plt.show()


if __name__ == "__main__":
    main()
