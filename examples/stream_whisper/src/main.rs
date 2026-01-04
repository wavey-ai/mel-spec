use mel_spec::prelude::*;
use mel_spec::vad::{duration_ms_for_n_frames, format_milliseconds};
use std::io::{self, Read};
use structopt::StructOpt;
use whisper_rs::*;

#[derive(Debug, StructOpt)]
#[structopt(name = "mel_spec_example", about = "mel_spec whisper example app")]
struct Command {
    #[structopt(
        short,
        long,
        default_value = "./../../../whisper.cpp/models/ggml-medium.en.bin"
    )]
    model_path: String,
    #[structopt(short, long, default_value = "./mel_out")]
    out_path: String,
    #[structopt(long, default_value = "1.0")]
    min_power: f64,
    #[structopt(long, default_value = "3")]
    min_y: usize,
    #[structopt(long, default_value = "5")]
    min_x: usize,
    #[structopt(long, default_value = "0")]
    min_mel: usize,
    #[structopt(long, default_value = "100")]
    min_frames: usize,
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
    let model_path = args.model_path;
    let mel_path = args.out_path;

    let min_power = args.min_power;
    let min_y = args.min_y;
    let min_x = args.min_x;
    let min_mel = args.min_mel;
    let min_frames = args.min_frames;

    let fft_size = 400;
    let hop_size = 160;
    let n_mels = 80;
    let sampling_rate = 16000.0;

    // Initialize processing stages
    let mut stft = Spectrogram::new(fft_size, hop_size);
    let mut mel = MelSpectrogram::new(fft_size, sampling_rate, n_mels);
    let vad_settings = DetectionSettings::new(min_power, min_y, min_x, min_mel);
    let mut vad = VoiceActivityDetector::new(&vad_settings);

    // Initialize whisper
    let ctx_params = WhisperContextParameters::default();
    let ctx = WhisperContext::new_with_params(&model_path, ctx_params).expect("failed to load model");
    let mut state = ctx.create_state().expect("failed to create state");

    // Buffer for accumulating mel frames
    let mut mel_frames: Vec<ndarray::Array2<f64>> = Vec::new();
    let mut frame_idx: usize = 0;

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
            frame_idx += 1;

            // Check VAD
            let is_speech = vad.add(&mel_frame);
            mel_frames.push(mel_frame);

            // Check if we have enough frames and hit a speech boundary
            if mel_frames.len() >= min_frames {
                if let Some(false) = is_speech {
                    // Non-speech boundary detected, process accumulated frames
                    let interleaved = interleave_frames(
                        &mel_frames,
                        false, // major row order for whisper
                        0,
                    );

                    // Save TGA for debugging
                    let path = format!("{}/frame_{}.tga", mel_path, frame_idx);
                    let _ = save_tga_8bit(&interleaved, n_mels, &path);

                    let ms = duration_ms_for_n_frames(hop_size, sampling_rate, frame_idx);
                    let time = format_milliseconds(ms as u64);

                    // Run whisper inference
                    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
                    params.set_n_threads(6);
                    params.set_single_segment(true);
                    params.set_language(Some("en"));
                    params.set_print_special(false);
                    params.set_print_progress(false);
                    params.set_print_realtime(false);
                    params.set_print_timestamps(false);

                    state.set_mel(&interleaved).unwrap();

                    let empty: Vec<f32> = vec![];
                    state.full(params, &empty[..]).unwrap();

                    let num_segments = state.full_n_segments().unwrap();
                    if num_segments > 0 {
                        if let Ok(text) = state.full_get_segment_text(0) {
                            let msg = format!("{} [{}] {}", frame_idx, time, text);
                            println!("{}", msg);
                        }
                    }

                    // Clear buffer, keep last few frames for context
                    mel_frames.clear();
                }
            }
        }
    }

    // Process any remaining frames
    if !mel_frames.is_empty() {
        let interleaved = interleave_frames(&mel_frames, false, 0);

        let ms = duration_ms_for_n_frames(hop_size, sampling_rate, frame_idx);
        let time = format_milliseconds(ms as u64);

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        params.set_n_threads(6);
        params.set_single_segment(true);
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state.set_mel(&interleaved).unwrap();

        let empty: Vec<f32> = vec![];
        state.full(params, &empty[..]).unwrap();

        let num_segments = state.full_n_segments().unwrap();
        if num_segments > 0 {
            if let Ok(text) = state.full_get_segment_text(0) {
                let msg = format!("{} [{}] {}", frame_idx, time, text);
                println!("{}", msg);
            }
        }
    }
}
