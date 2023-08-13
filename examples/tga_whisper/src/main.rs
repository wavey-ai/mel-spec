use mel_spec::prelude::*;
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
    #[structopt(short, long)]
    tga_path: String,
}

fn main() {
    let args = Command::from_args();
    let model_path = args.model_path;
    let tga_path = args.tga_path;

    let ctx = WhisperContext::new(&model_path).expect("failed to load model");
    let mut state = ctx.create_state().expect("failed to create key");
    let mel = load_tga_8bit(&tga_path).unwrap();

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    params.set_n_threads(1);
    params.set_single_segment(true);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    state.set_mel(&mel).unwrap();
    let empty = vec![];
    state.full(params, &empty[..]).unwrap();

    let num_segments = state.full_n_segments().unwrap();
    println!("Got {}", num_segments);

    if num_segments > 0 {
        if let Ok(text) = state.full_get_segment_text(0) {
            println!("{}", text);
        } else {
            println!("Error retrieving text for segment.");
        }
    }
}
