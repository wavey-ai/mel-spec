use mel_spec::prelude::*;
use mel_spec::vad::{duration_ms_for_n_frames, format_milliseconds};
use mel_spec_audio::packet::deinterleave_vecs_f32;
use mel_spec_pipeline::prelude::*;
use std::io::{self, Read};
use std::thread;
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

    let mel_settings = MelConfig::new(fft_size, hop_size, n_mels, sampling_rate);
    let vad_settings = DetectionSettings::new(min_power, min_y, min_x, min_mel);
    let audio_config = AudioConfig::new(32, 16000.0);

    let config = PipelineConfig::new(audio_config, mel_settings, vad_settings);

    let mut pipeline = Pipeline::new(config);

    let rx_clone = pipeline.mel_rx();
    let mut handles = pipeline.start();

    let handle = thread::spawn(move || {
        let ctx = WhisperContext::new(&model_path).expect("failed to load model");
        let mut state = ctx.create_state().expect("failed to create key");

        let mut buf = PipelineOutputBuffer::new();
        while let Ok(mel) = rx_clone.recv() {
            if let Some(frames) = buf.add(mel.idx(), mel.frame()) {
                let path = format!("{}/frame_{}.tga", mel_path, mel.idx());
                let _ = save_tga_8bit(&frames, n_mels, &path);

                let ms = duration_ms_for_n_frames(hop_size, sampling_rate, mel.idx());
                let time = format_milliseconds(ms as u64);

                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
                params.set_n_threads(6);
                params.set_single_segment(true);
                params.set_language(Some("en"));
                params.set_print_special(false);
                params.set_print_progress(false);
                params.set_print_realtime(false);
                params.set_print_timestamps(false);
                state.set_mel(&frames).unwrap();

                let empty = vec![];
                state.full(params, &empty[..]).unwrap();

                let num_segments = state.full_n_segments().unwrap();
                if num_segments > 0 {
                    if let Ok(text) = state.full_get_segment_text(0) {
                        let msg = format!("{} [{}] {}", mel.idx(), time, text);
                        println!("{}", msg);
                    } else {
                        println!("Error retrieving text for segment.");
                    }
                }
            }
        }
    });

    handles.push(handle);

    // read audio from pipe
    const LEN: usize = 128;
    let mut input: Box<dyn Read> = Box::new(io::stdin());

    let mut buffer = [0; LEN];
    let mut bytes_read = 0;
    loop {
        match input.read(&mut buffer[bytes_read..]) {
            Ok(0) => break,
            Ok(n) => bytes_read += n,
            Err(error) => {
                eprintln!("Error reading input: {}", error);
                std::process::exit(1);
            }
        }

        if bytes_read == LEN {
            let samples = deinterleave_vecs_f32(&buffer, 1);
            for chunk in samples[0].chunks(LEN / 4) {
                let _ = pipeline.send_pcm(chunk);
            }
            bytes_read = 0;
        }
    }

    if bytes_read > 0 {
        let samples = deinterleave_vecs_f32(&buffer[..bytes_read], 1);
        for chunk in samples[0].chunks(bytes_read / 4) {
            let _ = pipeline.send_pcm(chunk);
        }
    }

    pipeline.close_ingress();

    for handle in handles {
        handle.join().unwrap();
    }
}
