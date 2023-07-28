// Piepleine example using Stage functions

use crate::helpers::deinterleave_vecs_f32;
use crate::helpers::*;
use crate::mel::{interleave_frames, Stage as MelStage};
use crate::quant::{load_tga_8bit, save_tga_8bit};
use crate::stft::Stage as StftStage;
use crate::vad::{as_image, n_frames_for_duration, DetectionSettings, Stage as VadStage};
use crossbeam_channel::{bounded, unbounded, Receiver, RecvError, SendError, Sender};
use std::fs::File;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::thread::sleep;
use std::time::{Duration, Instant};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperError, WhisperState};

mod tests {
    use super::*;

    struct Config {
        // min number of ms to buffer sepctrograms before sending to processing stage
        bits_per_sample: usize,
        channel_count: usize,
        fft_size: usize,
        hop_size: usize,
        n_mels: usize,
        sampling_rate: f64,
        model_path: String,
        vad_settings: DetectionSettings,
    }

    struct Pipeline {
        // we drop data_rx by setting it to None, thereby removing all references to it.
        data_tx: Option<Sender<Vec<f32>>>,
        data_rx: Receiver<Vec<f32>>,
        config: Config,
    }

    impl Pipeline {
        fn new(config: Config) -> Self {
            let (data_tx, data_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();

            Self {
                data_tx: Some(data_tx),
                data_rx,
                config,
            }
        }

        pub fn send_data(&self, pcm: &[f32]) -> Result<(), SendError<Vec<f32>>> {
            self.data_tx
                .as_ref()
                .expect("not closed")
                .send(pcm.to_vec())
        }

        // to signal we are done streaming into the pipeline
        pub fn close_ingress(&mut self) {
            self.data_tx = None;
        }

        fn start(&self) -> Vec<thread::JoinHandle<()>> {
            // avoid cloning self
            let bits_per_sample = self.config.bits_per_sample;
            let bytes_per_sample = bits_per_sample / 8;
            let channel_count = self.config.channel_count;
            let fft_size = self.config.fft_size;
            let hop_size = self.config.hop_size;
            let model_path = self.config.model_path.to_string();
            let n_mels = self.config.n_mels;
            let sampling_rate = self.config.sampling_rate;
            let vad_settings = self.config.vad_settings;
            let buffer_len = bytes_per_sample * channel_count * fft_size;

            let (stt_tx, stt_rx) = mpsc::channel::<(usize, Vec<f32>)>();
            let stt_threads = 2;

            let mut handles = Vec::new();

            // Wrap the receiver in an Arc and Mutex for thread-safe sharing
            let stt_rx_shared = Arc::new(Mutex::new(stt_rx));

            // one possible approach. With CoreML (GPU) such cpu parallelisation wont help.
            // this demo effectivedly alternates threads / whisper processes each time
            for i in 0..stt_threads {
                let model_path_clone = model_path.clone();
                let rx_clone = Arc::clone(&stt_rx_shared);

                let handle = thread::spawn(move || {
                    let ctx = WhisperContext::new(&model_path_clone).expect("failed to load model");
                    let mut state = ctx.create_state().expect("failed to create key");

                    loop {
                        match rx_clone.lock().unwrap().try_recv() {
                            Ok((idx, mel)) => {
                                let ms = crate::vad::duration_ms_for_n_frames(
                                    hop_size,
                                    sampling_rate,
                                    idx,
                                );
                                let time = crate::vad::format_milliseconds(ms as u64);

                                let mut params =
                                    FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
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
                                if num_segments > 0 {
                                    if let Ok(text) = state.full_get_segment_text(0) {
                                        let msg = format!("({}) {} [{}] {}", i, idx, time, text);
                                        println!("{}", msg);
                                    } else {
                                        println!("Error retrieving text for segment.");
                                    }
                                }
                            }
                            Err(_) => {
                                // Handle the case when the channel is empty (end of data).
                                // Break out of the loop to terminate the worker thread.
                                break;
                            }
                        }
                        // Introduce a delay to avoid busy waiting
                        thread::sleep(Duration::from_millis(500));
                    }
                });

                handles.push(handle);
            }

            // internal channel to move decoded PCM audio to the fft processing stage
            // packet id, channel number, channel pcm f32 data
            let (pcm_tx, pcm_rx): (Sender<(usize, Vec<f32>)>, Receiver<(usize, Vec<f32>)>) =
                unbounded();

            let data_rx_clone = self.data_rx.clone();
            let pcm_rx_clone = pcm_rx.clone();

            // PCM -> Short-Time FFT -> Mel Spectrogram -> Voice-Activity Detection -> Speech-To-Text
            let fft_handle = thread::spawn(move || {
                let mut fft = StftStage::new(fft_size, hop_size);
                let mut mel = MelStage::new(fft_size, hop_size, sampling_rate, n_mels);
                let mut vad = VadStage::new(&vad_settings);

                let mut debug = Vec::new();
                let mut cutsecs = Vec::new();
                while let Ok((idx, samples)) = pcm_rx_clone.recv() {
                    // this will buffer up to an initial fft window
                    if let Some(complex) = fft.add(&samples) {
                        let spec = mel.add(complex);
                        // this will buffer up to an estimated word boundary
                        if let Some((idx, frames)) = vad.add(&spec) {
                            debug.extend_from_slice(&frames);
                            cutsecs.push(idx);

                            let data = interleave_frames(&frames.clone(), false, 200);
                            let path = format!("./test/vad/cutsec_{}.tga", idx);

                            // save the segment as spectrogram
                            save_tga_8bit(&data, n_mels, &path);

                            stt_tx.send((idx, data));
                        }
                    }
                }

                if let Some((idx, frames)) = vad.flush() {
                    stt_tx.send((idx, interleave_frames(&frames.clone(), false, 100)));
                }

                let data = interleave_frames(&debug.clone(), false, 100);

                let img = crate::vad::as_image(&debug, &cutsecs);
                img.save("./test/vad/cutsecs.png").unwrap();

                drop(stt_tx);
            });

            handles.push(fft_handle);

            // handler for external audio source.
            let pcm_handle = thread::spawn(move || {
                let mut accumulated_samples: Vec<f32> = Vec::new();
                let mut idx = 0;
                while let Ok(samples) = data_rx_clone.recv() {
                    accumulated_samples.extend_from_slice(&samples);
                    while accumulated_samples.len() >= hop_size {
                        let (chunk, rest) = accumulated_samples.split_at(hop_size);
                        pcm_tx.send((idx.clone(), chunk.to_vec())).unwrap();
                        idx = idx.wrapping_add(1);

                        accumulated_samples = rest.to_vec();
                    }
                }

                if !accumulated_samples.is_empty() {
                    pcm_tx
                        .send((idx.clone(), accumulated_samples.clone()))
                        .unwrap();
                }

                drop(pcm_tx);
            });

            handles.push(pcm_handle);

            handles
        }
    }

    #[test]
    fn test_pipeline() {
        // load the whisper jfk sample
        let file_path = "test/JFKWHA-001-AU_WR.wav";
        let file = File::open(&file_path).unwrap();
        let pcm = parse_wav(file).unwrap();
        assert_eq!(pcm.sampling_rate, 16000);
        assert_eq!(pcm.channel_count, 1);
        assert_eq!(pcm.bits_per_sample, 32);

        let samples = deinterleave_vecs_f32(&pcm.data, 1);

        let fft_size = 400;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16000.0;
        let channel_count = 1;
        let bits_per_sample = 32;
        let model_path = "./../whisper.cpp/models/ggml-medium.en.bin";
        let vad_settings = DetectionSettings {
            energy_threshold: 1.0,
            min_intersections: 4,
            intersection_threshold: 3,
            min_mel: 4,
            min_frames: 40,
        };
        let config = Config {
            vad_settings,
            n_mels,
            model_path: model_path.to_string(),
            fft_size,
            hop_size,
            sampling_rate,
            channel_count,
            bits_per_sample,
        };

        let mut pl = Pipeline::new(config);

        let handles = pl.start();

        for chunk in samples[0].chunks(88) {
            pl.send_data(chunk);
        }
        pl.close_ingress();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
