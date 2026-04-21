const PI: f32 = 3.14159265358979323846;

struct BitReverseUniforms {
    fft_size: u32,
    num_frames: u32,
    log2_size: u32,
    dispatch_offset: u32,
}

struct StageUniforms {
    fft_size: u32,
    num_frames: u32,
    stage_len: u32,
    half_len: u32,
    dispatch_offset: u32,
    _pad: u32,
}

struct MelUniforms {
    fft_size: u32,
    num_frames: u32,
    n_mels: u32,
    bins: u32,
    log10_scale: f32,
    epsilon: f32,
    dispatch_offset: u32,
    _pad: u32,
}

struct BluesteinPrepareUniforms {
    fft_size: u32,
    convolution_size: u32,
    num_frames: u32,
    dispatch_offset: u32,
}

struct PointwiseUniforms {
    fft_size: u32,
    num_frames: u32,
    dispatch_offset: u32,
    _pad: u32,
}

struct ConjugateUniforms {
    total_len: u32,
    dispatch_offset: u32,
    scale: f32,
    _pad: u32,
}

struct BluesteinPostUniforms {
    fft_size: u32,
    convolution_size: u32,
    num_frames: u32,
    dispatch_offset: u32,
    inverse_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> bitrev_input: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> bitrev_output: array<vec2<f32>>;
@group(0) @binding(2)
var<uniform> bitrev_uniforms: BitReverseUniforms;

@group(0) @binding(0)
var<storage, read> stage_input: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> stage_output: array<vec2<f32>>;
@group(0) @binding(2)
var<uniform> stage_uniforms: StageUniforms;

@group(0) @binding(0)
var<storage, read> mel_fft: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> mel_filters: array<f32>;
@group(0) @binding(2)
var<storage, read_write> mel_output: array<f32>;
@group(0) @binding(3)
var<uniform> mel_uniforms: MelUniforms;

@group(0) @binding(0)
var<storage, read> bluestein_prepare_input: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> bluestein_prepare_output: array<vec2<f32>>;
@group(0) @binding(2)
var<uniform> bluestein_prepare_uniforms: BluesteinPrepareUniforms;

@group(0) @binding(0)
var<storage, read> pointwise_left: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> pointwise_right: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read_write> pointwise_output: array<vec2<f32>>;
@group(0) @binding(3)
var<uniform> pointwise_uniforms: PointwiseUniforms;

@group(0) @binding(0)
var<storage, read> conjugate_input: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> conjugate_output: array<vec2<f32>>;
@group(0) @binding(2)
var<uniform> conjugate_uniforms: ConjugateUniforms;

@group(0) @binding(0)
var<storage, read> bluestein_post_input: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> bluestein_post_output: array<vec2<f32>>;
@group(0) @binding(2)
var<uniform> bluestein_post_uniforms: BluesteinPostUniforms;

fn reverse_bits(value: u32, bits: u32) -> u32 {
    var x = value;
    var out = 0u;
    for (var i = 0u; i < bits; i = i + 1u) {
        out = (out << 1u) | (x & 1u);
        x = x >> 1u;
    }
    return out;
}

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}

fn complex_conj(value: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(value.x, -value.y);
}

@compute @workgroup_size(64, 1, 1)
fn bitreverse_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = bitrev_uniforms.dispatch_offset + gid.x;
    let total = bitrev_uniforms.fft_size * bitrev_uniforms.num_frames;
    if (idx >= total) {
        return;
    }

    let frame = idx / bitrev_uniforms.fft_size;
    let lane = idx % bitrev_uniforms.fft_size;
    let src_lane = reverse_bits(lane, bitrev_uniforms.log2_size);
    let src_index = frame * bitrev_uniforms.fft_size + src_lane;
    bitrev_output[idx] = bitrev_input[src_index];
}

@compute @workgroup_size(64, 1, 1)
fn fft_stage_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let butterfly = stage_uniforms.dispatch_offset + gid.x;
    let butterflies_per_frame = stage_uniforms.fft_size / 2u;
    let total = stage_uniforms.num_frames * butterflies_per_frame;
    if (butterfly >= total) {
        return;
    }

    let frame = butterfly / butterflies_per_frame;
    let local = butterfly % butterflies_per_frame;
    let group = local / stage_uniforms.half_len;
    let pair = local % stage_uniforms.half_len;
    let i = group * stage_uniforms.stage_len + pair;
    let j = i + stage_uniforms.half_len;
    let base = frame * stage_uniforms.fft_size;

    let angle = -2.0 * PI * f32(pair) / f32(stage_uniforms.stage_len);
    let twiddle = vec2<f32>(cos(angle), sin(angle));
    let even = stage_input[base + i];
    let odd = complex_mul(stage_input[base + j], twiddle);

    stage_output[base + i] = even + odd;
    stage_output[base + j] = even - odd;
}

@compute @workgroup_size(64, 1, 1)
fn mel_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = mel_uniforms.dispatch_offset + gid.x;
    let total = mel_uniforms.num_frames * mel_uniforms.n_mels;
    if (idx >= total) {
        return;
    }

    let frame = idx / mel_uniforms.n_mels;
    let mel = idx % mel_uniforms.n_mels;
    let fft_base = frame * mel_uniforms.fft_size;
    let filter_base = mel * mel_uniforms.bins;
    let live_bins = mel_uniforms.fft_size / 2u;

    var sum = 0.0;
    for (var bin = 0u; bin < mel_uniforms.bins; bin = bin + 1u) {
        var magnitude = 0.0;
        if (bin < live_bins) {
            let value = mel_fft[fft_base + bin];
            magnitude = value.x * value.x + value.y * value.y;
        }
        sum = sum + magnitude * mel_filters[filter_base + bin];
    }

    mel_output[idx] = log(max(sum, mel_uniforms.epsilon)) * mel_uniforms.log10_scale;
}

@compute @workgroup_size(64, 1, 1)
fn bluestein_prepare_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = bluestein_prepare_uniforms.dispatch_offset + gid.x;
    let total = bluestein_prepare_uniforms.num_frames * bluestein_prepare_uniforms.convolution_size;
    if (idx >= total) {
        return;
    }

    let frame = idx / bluestein_prepare_uniforms.convolution_size;
    let lane = idx % bluestein_prepare_uniforms.convolution_size;
    if (lane >= bluestein_prepare_uniforms.fft_size) {
        bluestein_prepare_output[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let input_idx = frame * bluestein_prepare_uniforms.fft_size + lane;
    let lane_f = f32(lane);
    let angle = -PI * lane_f * lane_f / f32(bluestein_prepare_uniforms.fft_size);
    let chirp = vec2<f32>(cos(angle), sin(angle));
    bluestein_prepare_output[idx] = complex_mul(bluestein_prepare_input[input_idx], chirp);
}

@compute @workgroup_size(64, 1, 1)
fn pointwise_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = pointwise_uniforms.dispatch_offset + gid.x;
    let total = pointwise_uniforms.num_frames * pointwise_uniforms.fft_size;
    if (idx >= total) {
        return;
    }

    let lane = idx % pointwise_uniforms.fft_size;
    pointwise_output[idx] = complex_mul(pointwise_left[idx], pointwise_right[lane]);
}

@compute @workgroup_size(64, 1, 1)
fn conjugate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = conjugate_uniforms.dispatch_offset + gid.x;
    if (idx >= conjugate_uniforms.total_len) {
        return;
    }

    let value = complex_conj(conjugate_input[idx]);
    conjugate_output[idx] = value * conjugate_uniforms.scale;
}

@compute @workgroup_size(64, 1, 1)
fn bluestein_post_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = bluestein_post_uniforms.dispatch_offset + gid.x;
    let total = bluestein_post_uniforms.num_frames * bluestein_post_uniforms.fft_size;
    if (idx >= total) {
        return;
    }

    let frame = idx / bluestein_post_uniforms.fft_size;
    let bin = idx % bluestein_post_uniforms.fft_size;
    let conv_idx = frame * bluestein_post_uniforms.convolution_size + bin;
    let lane_f = f32(bin);
    let angle = -PI * lane_f * lane_f / f32(bluestein_post_uniforms.fft_size);
    let chirp = vec2<f32>(cos(angle), sin(angle));
    let value = complex_conj(bluestein_post_input[conv_idx]) * bluestein_post_uniforms.inverse_scale;
    bluestein_post_output[idx] = complex_mul(value, chirp);
}
