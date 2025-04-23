use rand::rng;
use rand_distr::{Distribution, Normal};

pub(crate) fn apply_dither(frame: &mut [f32], std: f32) {
    if std <= 0.0 {
        return;
    }
    let mut rn = rng();
    let noise = Normal::new(0.0, std).unwrap();
    for x in frame.iter_mut() {
        *x += noise.sample(&mut rn);
    }
}

pub(crate) fn apply_preemphasis(frame: &mut [f32], state: &mut f32, preemph: f32) {
    for sample in frame.iter_mut() {
        let x = *sample;
        let y = x - preemph * *state;
        *state = x;
        *sample = y;
    }
}
