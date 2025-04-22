#[derive(Clone)]
pub struct MelConfig {
    fft_size: usize,
    hop_size: usize,
    n_mels: usize,
    sampling_rate: f64,
}

impl MelConfig {
    pub fn new(fft_size: usize, hop_size: usize, n_mels: usize, sampling_rate: f64) -> Self {
        MelConfig {
            fft_size,
            hop_size,
            n_mels,
            sampling_rate,
        }
    }

    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    pub fn sampling_rate(&self) -> f64 {
        self.sampling_rate
    }
}
