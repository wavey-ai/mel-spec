use mel_spec::prelude::*;
use mel_spec_audio::deinterleave_vecs_f32;
use mel_spec_pipeline::pipeline::*;
use ndarray::Array2;
use std::fs::File;
use std::io::{self, Read, Write};
use std::thread;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "mel_tga", about = "mel_tga example app")]
struct Command {
    #[structopt(short, long, default_value = "80")]
    mels: usize,
    #[structopt(short, long, default_value = "./")]
    out_dir: String,
}

fn main() {
    let args = Command::from_args();
    let out_dir = args.out_dir;

    let fft_size = 400;
    let hop_size = 160;
    let n_mels = args.mels;
    let sampling_rate = 16000.0;
    let mel_settings = MelConfig::new(fft_size, hop_size, n_mels, sampling_rate);
    let vad_config = DetectionSettings::new(1.0, 3, 6, 0);
    let audio_config = AudioConfig::new(32, 16000.0);
    let config = PipelineConfig::new(audio_config, mel_settings, vad_config);

    let mut pipeline = Pipeline::new(config);

    let rx_clone = pipeline.mel_rx();
    let mut handles = pipeline.start();

    let handle = thread::spawn(move || {
        let mut mels = Vec::new();
        while let Ok((mel)) = rx_clone.recv() {
            mels.push(mel.frame().to_owned());
        }

        dbg!("saving");
        let frames = interleave_frames(&mels, false, 100);
        for (i, tga) in tga_8bit(&frames, n_mels).iter().enumerate() {
            let path = format!("{}/out_chunk{}.tga", out_dir, i);
            let mut file = File::create(path).unwrap();
            file.write_all(&tga).unwrap();
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
