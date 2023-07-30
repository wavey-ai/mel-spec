use std::io::Read;

pub struct AudioFileData {
    pub bits_per_sample: u8,
    pub channel_count: u8,
    pub data: Vec<u8>,
    pub sampling_rate: u32,
}

/// Parse a wav file
pub fn parse_wav<R: Read>(mut reader: R) -> Result<AudioFileData, String> {
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .map_err(|err| err.to_string())?;

    if &buffer[..4] != b"RIFF" || &buffer[8..12] != b"WAVE" {
        return Err("Not a WAV file".to_string());
    }

    let mut position = 12; // After "WAVE"

    while &buffer[position..position + 4] != b"fmt " {
        let chunk_size =
            u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
        position += chunk_size + 8; // Advance to next chunk
    }

    let fmt_chunk = &buffer[position..position + 24];
    let sampling_rate = u32::from_le_bytes(fmt_chunk[12..16].try_into().unwrap()) as u32;
    let bits_per_sample = u16::from_le_bytes(fmt_chunk[22..24].try_into().unwrap()) as u8;
    let channel_count = u16::from_le_bytes(fmt_chunk[10..12].try_into().unwrap()) as u8;

    // Move position to after "fmt " chunk
    let chunk_size =
        u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
    position += chunk_size + 8;

    while &buffer[position..position + 4] != b"data" {
        let chunk_size =
            u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
        position += chunk_size + 8; // Advance to next chunk
    }

    let data_chunk = buffer[position + 8..].to_vec(); // Skip "data" and size

    let result = AudioFileData {
        bits_per_sample,
        channel_count,
        sampling_rate,
        data: data_chunk,
    };

    Ok(result)
}

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
