mod helpers;
mod mel;
mod pipeline;
mod quant;
mod stft;
mod text;
mod vad;

/*
pub use helpers::*;
pub use mel::*;
pub use pipeline::*;
pub use quant::*;
pub use stft::*;
pub use text::*;
pub use vad::*;
pub use whisper:*;
*/
#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use ndarray_npy::NpzReader;
    use std::fs::File;

    use crate::assert_nearby;
    use crate::helpers::*;
    use crate::mel::{interleave_frames, log_mel_spectrogram, mel, norm_mel};
    use crate::quant::{load_tga_8bit, quantize, save_tga_8bit};
    use crate::stft::Stage as STFTStage;

    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

    fn round_vec_to_precision(vec: &mut Vec<f32>, precision: f32) {
        for value in vec.iter_mut() {
            *value = (*value / precision).round() * precision;
        }
    }
    #[test]
    fn test_mel() {
        // whisper mel filterbank
        let file_path = "test/mel_filters.npz";
        let f = File::open(file_path).unwrap();
        let mut npz = NpzReader::new(f).unwrap();
        let filters: Array2<f32> = npz.by_index(0).unwrap();
        let want: Array2<f64> = filters.mapv(|x| f64::from(x));
        let got = mel(16000.0, 400, 80, false, true);
        assert_eq!(got.shape(), vec![80, 201]);
        for i in 0..80 {
            assert_nearby!(got.row(i), want.row(i), 1.0e-7);
        }
    }

    /*
     *
     * 1) Load the jfk sample from wav into a mel spectrogram:
     *      this is now ready for inference, but its not really a spectrogram
     *      - it's a bunch of numbers.
     * 2) Quantize the mel spec into an 8 bit grayscale image and save as a TGA file.
     *      - this file is localed in text/quantized_mel.tga and can be opened directly
     *      in the OS.
     *      - it's a b&w spectrograph!
     *      But can we do inference on the actual image, even though it's now 8bit data?
     *  Let's try:
     * 3) Load the tga file back into a mel buffer, dequantising to f32.
     * 4) Run inference on the dequantised mel buffer.
     *
     */

    #[test]
    fn test_spec() {
        //let file_path = "./test/quantized_mel_golden.tga";
        let file_path = "./test/vad/cutsec_41026.tga";
        let mut dequantized_mel = load_tga_8bit(file_path).unwrap();
        let ctx =
            WhisperContext::new("/Users/jamieb/wavey.ai/whisper.cpp/models/ggml-medium.en.bin")
                .expect("failed to load model");
        let mut state = ctx.create_state().expect("failed to create key");

        dbg!(&dequantized_mel.len());
        // set the spectrogram directly to the whisper state
        state.set_mel(&dequantized_mel).unwrap();

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        params.set_n_threads(4);
        params.set_single_segment(true);
        params.set_language(Some("en"));
        params.set_print_special(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // empty audio - whisper.cpp won't overwrite the mel state unless there are audio samples.
        let empty = vec![0.0; 0];

        state.full(params, &empty[..]).unwrap();
        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");
        assert_eq!(num_segments, 1);

        let got = state
            .full_get_segment_text(0)
            .expect("failed to get segment");
        assert_eq!(got, "[_BEG_] And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.[_TT_550]");
    }
}
