import numpy as np
import torch
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

featurizer = FilterbankFeatures(
    sample_rate=16000,
    n_window_size=int(0.02 * 16000),
    n_window_stride=int(0.01 * 16000),
    nfilt=80,
    window="hann",
    normalize="per_feature",
    n_fft=None,
    preemph=0.97,
    lowfreq=0,
    highfreq=None,
    log=True,
    log_zero_guard_type="add",
    log_zero_guard_value=2**-24,
    dither=1e-5,
    pad_to=16,
    frame_splicing=1,
    exact_pad=False,
    pad_value=0,
    mag_power=2.0,
    nb_augmentation_prob=0.0,
    nb_max_freq=4000,
    mel_norm="slaney",
    stft_exact_pad=False,
    stft_conv=False,
)

banks = featurizer.filter_banks.cpu().numpy()
np.savez("testdata/nemo_mel_filters.npz", banks=banks)
print(f"Saved nemo_mel_filters.npz with shape {banks.shape}")
