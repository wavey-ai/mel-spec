/// Split channel-interleaved audio into separate vectors each with non-interleaved audio
/// for that channel
pub fn deinterleave_vecs_f32(input: &[u8], channel_count: usize) -> Vec<Vec<f32>> {
    let sample_size = input.len() / (channel_count * 4);
    let mut result = vec![vec![0.0; sample_size]; channel_count];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let start = (i * channel_count + channel) * 4;
            let value = f32::from_le_bytes(input[start..start + 4].try_into().unwrap());
            result[channel][i] = value;
        }
    }

    result
}
