use std::convert::TryInto;
use std::io::{Error, Read};

pub struct AudioFileData {
    bits_per_sample: u8,
    channel_count: u8,
    sampling_rate: u32,
    data: Vec<u8>,
}

enum StreamWavState {
    Initial,
    ReadingFmt,
    ReadingData,
    Finished,
}

pub struct WavStreamProcessor {
    state: StreamWavState,
    buffer: Vec<u8>,
}

impl WavStreamProcessor {
    pub fn new() -> Self {
        Self {
            state: StreamWavState::Initial,
            buffer: Vec::new(),
        }
    }

    pub fn process_chunk(&mut self, chunk: &[u8]) -> Result<Option<AudioFileData>, String> {
        self.buffer.extend(chunk);

        let mut sampling_rate: u32 = 0;
        let mut bits_per_sample: u8 = 0;
        let mut channel_count: u8 = 0;

        loop {
            match &self.state {
                StreamWavState::Initial => {
                    if self.buffer.len() < 12 {
                        return Ok(None); // Wait for more data
                    }

                    if &self.buffer[..4] != b"RIFF" || &self.buffer[8..12] != b"WAVE" {
                        return Err("Not a WAV file".to_string());
                    }

                    self.state = StreamWavState::ReadingFmt;
                }

                StreamWavState::ReadingFmt => {
                    if self.buffer.len() < 16 {
                        return Ok(None); // Wait for more data
                    }

                    let chunk_size =
                        u32::from_le_bytes(self.buffer[4..8].try_into().unwrap()) as usize;
                    if self.buffer.len() < chunk_size + 8 {
                        return Ok(None); // Wait for more data
                    }

                    let fmt_chunk = &self.buffer[0..chunk_size + 8];
                    sampling_rate =
                        u32::from_le_bytes(fmt_chunk[12..16].try_into().unwrap()) as u32;
                    bits_per_sample =
                        u16::from_le_bytes(fmt_chunk[22..24].try_into().unwrap()) as u8;
                    channel_count = u16::from_le_bytes(fmt_chunk[10..12].try_into().unwrap()) as u8;

                    self.buffer.drain(0..chunk_size + 8);
                    self.state = StreamWavState::ReadingData;
                }

                StreamWavState::ReadingData => {
                    if self.buffer.len() < 8 {
                        return Ok(None); // Wait for more data
                    }

                    let chunk_size =
                        u32::from_le_bytes(self.buffer[4..8].try_into().unwrap()) as usize;
                    if self.buffer.len() < chunk_size + 8 {
                        return Ok(None); // Wait for more data
                    }

                    let data_chunk = self.buffer[8..chunk_size + 8].to_vec();

                    self.buffer.drain(0..chunk_size + 8);
                    self.state = StreamWavState::Finished;

                    let result = AudioFileData {
                        bits_per_sample,
                        channel_count,
                        sampling_rate,
                        data: data_chunk,
                    };

                    return Ok(Some(result));
                }

                StreamWavState::Finished => {
                    return Err("Already finished processing WAV file".to_string());
                }
            }
        }
    }
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
