#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum EncodingFlag {
    PCM = 0,
    Opus = 1,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AudioConfig {
    Hz16000Bit16,
    Hz16000Bit24,
    Hz16000Bit32,
    Hz44100Bit16,
    Hz44100Bit24,
    Hz44100Bit32,
    Hz48000Bit16,
    Hz48000Bit24,
    Hz48000Bit32,
    Hz88200Bit16,
    Hz88200Bit24,
    Hz88200Bit32,
    Hz96000Bit16,
    Hz96000Bit24,
    Hz96000Bit32,
    Hz176400Bit16,
    Hz176400Bit24,
    Hz176400Bit32,
    Hz192000Bit16,
    Hz192000Bit24,
    Hz192000Bit32,
    Hz352800Bit16,
    Hz352800Bit24,
    Hz352800Bit32,
}

pub fn get_sampling_rate_and_bits_per_sample(config: AudioConfig) -> (u32, u8) {
    match config {
        AudioConfig::Hz44100Bit16 => (16000, 16),
        AudioConfig::Hz44100Bit24 => (16000, 24),
        AudioConfig::Hz44100Bit32 => (16000, 32),
        AudioConfig::Hz44100Bit16 => (44100, 16),
        AudioConfig::Hz44100Bit24 => (44100, 24),
        AudioConfig::Hz44100Bit32 => (44100, 32),
        AudioConfig::Hz48000Bit16 => (48000, 16),
        AudioConfig::Hz48000Bit24 => (48000, 24),
        AudioConfig::Hz48000Bit32 => (48000, 32),
        AudioConfig::Hz88200Bit16 => (88200, 16),
        AudioConfig::Hz88200Bit24 => (88200, 24),
        AudioConfig::Hz88200Bit32 => (88200, 32),
        AudioConfig::Hz96000Bit16 => (96000, 16),
        AudioConfig::Hz96000Bit24 => (96000, 24),
        AudioConfig::Hz96000Bit32 => (96000, 32),
        AudioConfig::Hz176400Bit16 => (176400, 16),
        AudioConfig::Hz176400Bit24 => (176400, 24),
        AudioConfig::Hz176400Bit32 => (176400, 32),
        AudioConfig::Hz192000Bit16 => (192000, 16),
        AudioConfig::Hz192000Bit24 => (192000, 24),
        AudioConfig::Hz192000Bit32 => (192000, 32),
        AudioConfig::Hz352800Bit16 => (352800, 16),
        AudioConfig::Hz352800Bit24 => (352800, 24),
        AudioConfig::Hz352800Bit32 => (352800, 32),
    }
}

pub fn get_config(id: u8) -> AudioConfig {
    match id {
        0 => AudioConfig::Hz44100Bit16,
        1 => AudioConfig::Hz44100Bit24,
        2 => AudioConfig::Hz44100Bit32,
        3 => AudioConfig::Hz48000Bit16,
        4 => AudioConfig::Hz48000Bit24,
        5 => AudioConfig::Hz48000Bit32,
        6 => AudioConfig::Hz88200Bit16,
        7 => AudioConfig::Hz88200Bit24,
        8 => AudioConfig::Hz88200Bit32,
        9 => AudioConfig::Hz96000Bit16,
        10 => AudioConfig::Hz96000Bit24,
        11 => AudioConfig::Hz96000Bit32,
        12 => AudioConfig::Hz176400Bit16,
        13 => AudioConfig::Hz176400Bit24,
        14 => AudioConfig::Hz176400Bit32,
        15 => AudioConfig::Hz192000Bit16,
        16 => AudioConfig::Hz192000Bit24,
        17 => AudioConfig::Hz192000Bit32,
        18 => AudioConfig::Hz352800Bit16,
        19 => AudioConfig::Hz352800Bit24,
        20 => AudioConfig::Hz352800Bit32,
        _ => panic!("Invalid config ID"),
    }
}

pub fn get_audio_config(sampling_rate: u32, bits_per_sample: u8) -> Result<AudioConfig, String> {
    match (sampling_rate, bits_per_sample) {
        (16000, 16) => Ok(AudioConfig::Hz44100Bit16),
        (16000, 24) => Ok(AudioConfig::Hz44100Bit24),
        (16000, 32) => Ok(AudioConfig::Hz44100Bit32),
        (44100, 16) => Ok(AudioConfig::Hz44100Bit16),
        (44100, 24) => Ok(AudioConfig::Hz44100Bit24),
        (44100, 32) => Ok(AudioConfig::Hz44100Bit32),
        (48000, 16) => Ok(AudioConfig::Hz48000Bit16),
        (48000, 24) => Ok(AudioConfig::Hz48000Bit24),
        (48000, 32) => Ok(AudioConfig::Hz48000Bit32),
        (88200, 16) => Ok(AudioConfig::Hz88200Bit16),
        (88200, 24) => Ok(AudioConfig::Hz88200Bit24),
        (88200, 32) => Ok(AudioConfig::Hz88200Bit32),
        (96000, 16) => Ok(AudioConfig::Hz96000Bit16),
        (96000, 24) => Ok(AudioConfig::Hz96000Bit24),
        (96000, 32) => Ok(AudioConfig::Hz96000Bit32),
        (176400, 16) => Ok(AudioConfig::Hz176400Bit16),
        (176400, 24) => Ok(AudioConfig::Hz176400Bit24),
        (176400, 32) => Ok(AudioConfig::Hz176400Bit32),
        (192000, 16) => Ok(AudioConfig::Hz192000Bit16),
        (192000, 24) => Ok(AudioConfig::Hz192000Bit24),
        (192000, 32) => Ok(AudioConfig::Hz192000Bit32),
        (352800, 16) => Ok(AudioConfig::Hz352800Bit16),
        (352800, 24) => Ok(AudioConfig::Hz352800Bit24),
        (352800, 32) => Ok(AudioConfig::Hz352800Bit32),
        _ => Err(format!(
            "Unsupported audio configuration: sampling_rate={} bits_per_sample={}",
            sampling_rate, bits_per_sample
        )),
    }
}
