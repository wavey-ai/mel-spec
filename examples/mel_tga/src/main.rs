use mel_spec::prelude::*;
use std::fs::File;
use std::io::{self, Read, Write};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "mel_tga", about = "mel_tga example app")]
struct Command {
    #[structopt(short, long, default_value = "80")]
    mels: usize,
    #[structopt(short, long, default_value = "./")]
    out_dir: String,
}

/// Deinterleave bytes to f32 samples (mono, little-endian f32)
fn bytes_to_f32_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn main() {
    let args = Command::from_args();
    let out_dir = args.out_dir;

    let fft_size = 400;
    let hop_size = 160;
    let n_mels = args.mels;
    let sampling_rate = 16000.0;

    // Initialize processing stages
    let mut stft = Spectrogram::new(fft_size, hop_size);
    let mut mel = MelSpectrogram::new(fft_size, sampling_rate, n_mels);

    // Buffer for accumulating mel frames
    let mut mel_frames: Vec<ndarray::Array2<f64>> = Vec::new();

    // Read audio from stdin
    const CHUNK_SIZE: usize = 640; // 160 samples * 4 bytes per f32
    let mut input: Box<dyn Read> = Box::new(io::stdin());
    let mut buffer = vec![0u8; CHUNK_SIZE];

    loop {
        match input.read_exact(&mut buffer) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                std::process::exit(1);
            }
        }

        let samples = bytes_to_f32_samples(&buffer);

        // Process through STFT
        if let Some(fft_frame) = stft.add(&samples) {
            // Process through Mel filterbank
            let mel_frame = mel.add(&fft_frame);
            mel_frames.push(mel_frame);
        }
    }

    // Save output as TGA files
    if !mel_frames.is_empty() {
        eprintln!("Saving {} mel frames", mel_frames.len());
        let frames = interleave_frames(&mel_frames, false, 100);
        for (i, tga) in tga_8bit(&frames, n_mels).iter().enumerate() {
            let path = format!("{}/out_chunk{}.tga", out_dir, i);
            let mut file = File::create(&path).unwrap();
            file.write_all(tga).unwrap();
            eprintln!("Wrote {}", path);
        }
    }
}
