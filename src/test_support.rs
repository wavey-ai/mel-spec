//! Seeded random inputs for the differential fuzz tests.
//!
//! Set `MEL_SPEC_FUZZ_SCALE` to multiply the number of cases, for example
//! `MEL_SPEC_FUZZ_SCALE=50 cargo test --release fuzz_`.

/// xorshift64* generator. A fixed seed makes each failure reproducible.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform integer in `lo..=hi`.
    pub(crate) fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
    }

    /// Uniform value in `[0, 1)`.
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    pub(crate) fn chance(&mut self, probability: f64) -> bool {
        self.unit() < probability
    }

    /// Audio made of random segments: silence, tones, noise, impulses, DC,
    /// subnormal values, and loud values.
    pub(crate) fn signal(&mut self, len: usize) -> Vec<f32> {
        let mut samples = Vec::with_capacity(len);
        while samples.len() < len {
            let segment = self.range(1, 400).min(len - samples.len());
            let kind = self.range(0, 6);
            let amplitude = [1e-4, 0.01, 0.5, 1.0, 1e3][self.range(0, 4)];
            let freq = self.unit() * 0.5;
            let phase = self.unit() * std::f64::consts::TAU;
            for idx in 0..segment {
                let value = match kind {
                    0 => 0.0,
                    1 => amplitude * (phase + std::f64::consts::TAU * freq * idx as f64).sin(),
                    2 => amplitude * (self.unit() * 2.0 - 1.0),
                    3 if idx == 0 => amplitude,
                    3 => 0.0,
                    4 => amplitude,
                    5 => (self.unit() * 2.0 - 1.0) * 1e-40,
                    _ => amplitude * (self.unit() * 2.0 - 1.0) * (idx % 7) as f64,
                };
                samples.push(value as f32);
            }
        }
        samples
    }
}

/// Number of cases for a fuzz test: `default` times `MEL_SPEC_FUZZ_SCALE`.
pub(crate) fn fuzz_cases(default: usize) -> usize {
    let scale = std::env::var("MEL_SPEC_FUZZ_SCALE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    default * scale.max(1)
}

/// Equal values, with NaN equal to NaN.
pub(crate) fn same_f32(a: f32, b: f32) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

pub(crate) fn same_f64(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}
