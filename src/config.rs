use crate::pcm::{apply_dither, apply_preemphasis};
use crate::{config::MelConfig as _, stft};
use ndarray::{concatenate, s, Array2, Axis};
use rtrb::{Consumer, PopError, Producer, PushError, RingBuffer as Rtrb};
use std::time::Instant;

#[derive(Debug, Clone)]
pub enum Window {
    Hann,
}

#[derive(Clone, Copy, Debug)]
pub enum LogType {
    Ln,
    Log10,
}

#[derive(Debug, Clone)]
pub enum Normalize {
    PerFeature,
    AllFeature,
}

#[derive(Debug, Clone)]
pub enum LogZeroGuardType {
    Add,
}

#[derive(Debug, Clone)]
pub enum MelNorm {
    Slaney,
}

pub trait MelConfig {
    fn sample_rate(&self) -> f64;
    fn n_fft(&self) -> usize;
    fn n_window_size(&self) -> usize;
    fn n_window_stride(&self) -> usize;
    fn features(&self) -> usize;
    fn window(&self) -> Window;
    fn normalize(&self) -> Normalize;
    fn mel_norm(&self) -> MelNorm;
    fn mag_power(&self) -> f32;
    fn preemph(&self) -> Option<f32>;
    fn dither(&self) -> Option<f32>;
    fn log_type(&self) -> LogType;
    fn log_zero_guard_type(&self) -> LogZeroGuardType;
    fn log_zero_guard_value(&self) -> f32;
    fn lowfreq(&self) -> f32;
    fn highfreq(&self) -> Option<f32>;
    fn pad_to(&self) -> usize;
    fn pad_value(&self) -> f32;
    fn exact_pad(&self) -> bool;
    fn frame_splicing(&self) -> usize;
    fn max_duration(&self) -> f32;
    fn nb_augmentation_prob(&self) -> f32;
    fn nb_max_freq(&self) -> usize;
    fn rng(&self) -> Option<u64>;
    fn stft_conv(&self) -> bool;
    fn stft_exact_pad(&self) -> bool;
}

// shared common defaults
#[derive(Debug, Clone)]
pub struct CommonConfig {
    pub sample_rate: f64,
    pub n_fft: usize,
    pub n_window_size: usize,
    pub n_window_stride: usize,
    pub features: usize,
    pub window: Window,
    pub normalize: Normalize,
    pub mel_norm: MelNorm,
    pub mag_power: f32,
    pub preemph: Option<f32>,
    pub dither: Option<f32>,
}

// NeMo preset
#[derive(Debug, Clone)]
pub struct NemoConfig {
    pub common: CommonConfig,
    pub exact_pad: bool,
    pub pad_to: usize,
    pub pad_value: f32,
    pub lowfreq: f32,
    pub highfreq: Option<f32>,
    pub log_type: LogType,
    pub log_zero_guard_type: LogZeroGuardType,
    pub log_zero_guard_value: f32,
    pub frame_splicing: usize,
    pub max_duration: f32,
    pub nb_augmentation_prob: f32,
    pub nb_max_freq: usize,
    pub rng: Option<u64>,
    pub stft_conv: bool,
    pub stft_exact_pad: bool,
}

impl Default for NemoConfig {
    fn default() -> Self {
        Self {
            common: CommonConfig {
                sample_rate: 16000.0,
                n_fft: 512,
                n_window_size: 400,
                n_window_stride: 160,
                features: 80,
                window: Window::Hann,
                normalize: Normalize::PerFeature,
                mel_norm: MelNorm::Slaney,
                mag_power: 2.0,
                preemph: Some(0.97),
                dither: Some(1e-5),
            },
            exact_pad: false,
            pad_to: 16,
            pad_value: 0.0,
            lowfreq: 0.0,
            highfreq: None,
            log_type: LogType::Ln,
            log_zero_guard_type: LogZeroGuardType::Add,
            log_zero_guard_value: 2_f32.powi(-24),
            frame_splicing: 1,
            max_duration: 16.7,
            nb_augmentation_prob: 0.0,
            nb_max_freq: 4000,
            rng: None,
            stft_conv: false,
            stft_exact_pad: false,
        }
    }
}

impl MelConfig for NemoConfig {
    fn sample_rate(&self) -> f64 {
        self.common.sample_rate
    }
    fn n_fft(&self) -> usize {
        self.common.n_fft
    }
    fn n_window_size(&self) -> usize {
        self.common.n_window_size
    }
    fn n_window_stride(&self) -> usize {
        self.common.n_window_stride
    }
    fn features(&self) -> usize {
        self.common.features
    }
    fn window(&self) -> Window {
        self.common.window.clone()
    }
    fn normalize(&self) -> Normalize {
        self.common.normalize.clone()
    }
    fn mel_norm(&self) -> MelNorm {
        self.common.mel_norm.clone()
    }
    fn mag_power(&self) -> f32 {
        self.common.mag_power
    }
    fn preemph(&self) -> Option<f32> {
        self.common.preemph
    }
    fn dither(&self) -> Option<f32> {
        self.common.dither
    }
    fn log_type(&self) -> LogType {
        self.log_type
    }
    fn log_zero_guard_type(&self) -> LogZeroGuardType {
        self.log_zero_guard_type.clone()
    }
    fn log_zero_guard_value(&self) -> f32 {
        self.log_zero_guard_value
    }
    fn lowfreq(&self) -> f32 {
        self.lowfreq
    }
    fn highfreq(&self) -> Option<f32> {
        self.highfreq
    }
    fn pad_to(&self) -> usize {
        self.pad_to
    }
    fn pad_value(&self) -> f32 {
        self.pad_value
    }
    fn exact_pad(&self) -> bool {
        self.exact_pad
    }
    fn frame_splicing(&self) -> usize {
        self.frame_splicing
    }
    fn max_duration(&self) -> f32 {
        self.max_duration
    }
    fn nb_augmentation_prob(&self) -> f32 {
        self.nb_augmentation_prob
    }
    fn nb_max_freq(&self) -> usize {
        self.nb_max_freq
    }
    fn rng(&self) -> Option<u64> {
        self.rng
    }
    fn stft_conv(&self) -> bool {
        self.stft_conv
    }
    fn stft_exact_pad(&self) -> bool {
        self.stft_exact_pad
    }
}

// Whisper preset
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub common: CommonConfig,
    pub log_type: LogType,
    pub log_zero_guard_type: LogZeroGuardType,
    pub log_zero_guard_value: f32,
    pub top_db: f32,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            common: CommonConfig {
                sample_rate: 16000.0,
                n_fft: 400,
                n_window_size: 400,
                n_window_stride: 160,
                features: 80,
                window: Window::Hann,
                normalize: Normalize::PerFeature,
                mel_norm: MelNorm::Slaney,
                mag_power: 2.0,
                preemph: None,
                dither: None,
            },
            log_type: LogType::Log10,
            log_zero_guard_type: LogZeroGuardType::Add,
            log_zero_guard_value: 1e-10,
            top_db: 8.0,
        }
    }
}

impl MelConfig for WhisperConfig {
    fn sample_rate(&self) -> f64 {
        self.common.sample_rate
    }
    fn n_fft(&self) -> usize {
        self.common.n_fft
    }
    fn n_window_size(&self) -> usize {
        self.common.n_window_size
    }
    fn n_window_stride(&self) -> usize {
        self.common.n_window_stride
    }
    fn features(&self) -> usize {
        self.common.features
    }
    fn window(&self) -> Window {
        self.common.window.clone()
    }
    fn normalize(&self) -> Normalize {
        self.common.normalize.clone()
    }
    fn mel_norm(&self) -> MelNorm {
        self.common.mel_norm.clone()
    }
    fn mag_power(&self) -> f32 {
        self.common.mag_power
    }
    fn preemph(&self) -> Option<f32> {
        self.common.preemph
    }
    fn dither(&self) -> Option<f32> {
        self.common.dither
    }
    fn log_type(&self) -> LogType {
        self.log_type
    }
    fn log_zero_guard_type(&self) -> LogZeroGuardType {
        self.log_zero_guard_type.clone()
    }
    fn log_zero_guard_value(&self) -> f32 {
        self.log_zero_guard_value
    }
    fn lowfreq(&self) -> f32 {
        0.0
    }
    fn highfreq(&self) -> Option<f32> {
        None
    }
    fn pad_to(&self) -> usize {
        0
    }
    fn pad_value(&self) -> f32 {
        0.0
    }
    fn exact_pad(&self) -> bool {
        false
    }
    fn frame_splicing(&self) -> usize {
        1
    }
    fn max_duration(&self) -> f32 {
        0.0
    }
    fn nb_augmentation_prob(&self) -> f32 {
        0.0
    }
    fn nb_max_freq(&self) -> usize {
        0
    }
    fn rng(&self) -> Option<u64> {
        None
    }
    fn stft_conv(&self) -> bool {
        false
    }
    fn stft_exact_pad(&self) -> bool {
        false
    }
}
