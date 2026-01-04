use mel_spec::prelude::*;
use structopt::StructOpt;
use whisper_rs::*;

#[derive(Debug, StructOpt)]
#[structopt(name = "tga_whisper", about = "Transcribe from a TGA mel spectrogram")]
struct Command {
    #[structopt(
        short,
        long,
        default_value = "./../../../whisper.cpp/models/ggml-medium.en.bin"
    )]
    model_path: String,
    #[structopt(short, long)]
    tga_path: String,
}

fn main() {
    let args = Command::from_args();
    let model_path = args.model_path;
    let tga_path = args.tga_path;

    let ctx_params = WhisperContextParameters::default();
    let ctx =
        WhisperContext::new_with_params(&model_path, ctx_params).expect("failed to load model");
    let mut state = ctx.create_state().expect("failed to create state");
    let mel = load_tga_8bit(&tga_path).unwrap();

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    params.set_n_threads(1);
    params.set_single_segment(true);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Set the pre-computed mel spectrogram and run inference with empty samples.
    // This workflow is supported by whisper.cpp PR #1214 and our whisper-rs fork.
    state.set_mel(&mel).unwrap();
    state.full(params, &[]).unwrap();

    let num_segments = state.full_n_segments().unwrap();
    println!("Got {} segments", num_segments);

    for i in 0..num_segments {
        if let Ok(text) = state.full_get_segment_text(i) {
            println!("{}", text);
        }
    }
}
