use crossbeam_channel::{unbounded, Receiver, SendError, Sender};
use mel_spec::prelude::*;
use ndarray::Array2;
use std::sync::{Arc, Mutex};
use std::thread;

pub struct PipelineConfig {
    vad_settings: Option<DetectionSettings>,
    mel_settings: MelConfig,
}

impl PipelineConfig {
    pub fn new(mel_settings: MelConfig, vad_settings: Option<DetectionSettings>) -> Self {
        Self {
            mel_settings,
            vad_settings,
        }
    }
}

pub struct Pipeline {
    // we drop data_tx by setting it to None, thereby removing all references to it.
    data_tx: Option<Sender<Vec<f32>>>,
    data_rx: Receiver<Vec<f32>>,
    stt_tx: Arc<Mutex<Option<Sender<(usize, Array2<f64>)>>>>,
    stt_rx: Receiver<(usize, Array2<f64>)>,
    config: PipelineConfig,
}

/// A simple pipeline for streaming audio in and getting speech-boundary delimited
/// mel spectrograms out.
///
/// Processing audio to 32f mono, and sending mel spectrograms to Speech-to-Text
/// would be separate input and output stages decoupled from this pipeline.
/// See examples folder.
impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let (data_tx, data_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();
        let (stt_tx, stt_rx): (Sender<(usize, Array2<f64>)>, Receiver<(usize, Array2<f64>)>) =
            unbounded();

        let stt_tx_arc = Arc::new(Mutex::new(Some(stt_tx)));
        Self {
            data_tx: Some(data_tx),
            data_rx,
            stt_tx: stt_tx_arc,
            stt_rx,
            config,
        }
    }

    /// send PCM mono, non-interleaved f32 samples - chunks can be any length.
    pub fn send_pcm(&self, pcm: &[f32]) -> Result<(), SendError<Vec<f32>>> {
        self.data_tx
            .as_ref()
            .expect("not closed")
            .send(pcm.to_vec())
    }

    /// receive frame index and spectrogram
    pub fn rx(&self) -> Receiver<(usize, Array2<f64>)> {
        self.stt_rx.clone()
    }

    /// to signal we are done streaming into the pipeline
    pub fn close_ingress(&mut self) {
        self.data_tx = None;
    }

    /// start background threads
    pub fn start(&mut self) -> Vec<thread::JoinHandle<()>> {
        // avoid cloning self
        let fft_size = self.config.mel_settings.fft_size();
        let hop_size = self.config.mel_settings.hop_size();
        let n_mels = self.config.mel_settings.n_mels();
        let sampling_rate = self.config.mel_settings.sampling_rate();
        let vad_settings = self.config.vad_settings;
        let mut handles = Vec::new();

        // internal channel to move decoded PCM audio to the fft processing stage
        // packet idx, pcm f32 data
        let (pcm_tx, pcm_rx): (Sender<(usize, Vec<f32>)>, Receiver<(usize, Vec<f32>)>) =
            unbounded();

        let (fft_tx, fft_rx): (Sender<(usize, Array2<f64>)>, Receiver<(usize, Array2<f64>)>) =
            unbounded();

        let data_rx_clone = self.data_rx.clone();
        let stt_tx_clone = Arc::clone(&self.stt_tx);

        // PCM -> Short-Time FFT -> Mel Spectrogram -> Voice-Activity Detection -> Speech-To-Text?
        let fft_handle = thread::spawn(move || {
            let mut fft = Spectrogram::new(fft_size, hop_size);
            let mut mel = MelSpectrogram::new(fft_size, sampling_rate, n_mels);
            let mut settings = DetectionSettings::default();
            if let Some(vad_settings) = vad_settings {
                settings = vad_settings;
            }

            let mut vad = VoiceActivityDetector::new(&settings);

            let mut idx = 0;
            while let Ok((_, samples)) = pcm_rx.recv() {
                // buffer up to an initial fft window
                if let Some(complex) = fft.add(&samples) {
                    let spec = mel.add(complex);
                    if let Some(_) = vad_settings {
                        // buffer up to speech boundary
                        if let Some((idx, frames)) = vad.add(&spec) {
                            for frame in frames {
                                if let Err(send_error) = fft_tx.send((idx, frame)) {
                                    eprintln!("Failed to send message to fft: {:?}", send_error);
                                }
                            }
                        }
                    } else {
                        if let Err(send_error) = fft_tx.send((idx, spec)) {
                            eprintln!("Failed to send message to fft: {:?}", send_error);
                        }
                        idx += 1;
                    }
                }
            }

            if let Some((idx, frames)) = vad.flush() {
                for frame in frames {
                    if let Err(send_error) = fft_tx.send((idx, frame)) {
                        // Handle the error here
                        eprintln!("Failed to send message to stt: {:?}", send_error);
                    }
                }
            }

            drop(fft_tx);
        });

        handles.push(fft_handle);

        let stt_handle = thread::spawn(move || {
            while let Ok((idx, frames)) = fft_rx.recv() {
                if let Some(stt_tx) = stt_tx_clone.lock().unwrap().as_ref() {
                    if let Err(send_error) = stt_tx.send((idx, frames)) {
                        eprintln!("Failed to send message to stt: {:?}", send_error);
                    }
                }
            }
            if let Some(_) = stt_tx_clone.lock().unwrap().take() {
                // At this point, stt_tx goes out of scope and will be dropped
                // The lock is automatically released when stt_tx goes out of scope
            }
        });

        handles.push(stt_handle);

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

pub struct PipelineOutputBuffer {
    buffer: Vec<Array2<f64>>,
    idx: usize,
}

impl PipelineOutputBuffer {
    pub fn new() -> Self {
        Self {
            idx: 0,
            buffer: Vec::new(),
        }
    }

    pub fn add(&mut self, idx: usize, frame: Array2<f64>) -> Option<Vec<f32>> {
        if idx != self.idx {
            let window = self.buffer.clone();
            self.buffer.drain(..);
            self.idx = idx;
            let frames = interleave_frames(&window, false, 100);
            return Some(frames);
        } else {
            self.buffer.push(frame);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mel_spec_audio::*;
    use std::fs::File;

    #[test]
    fn test_pipeline() {
        // load the whisper jfk sample
        let file_path = "../testdata/jfk_f32le.wav";
        let file = File::open(&file_path).unwrap();
        let data = parse_wav(file).unwrap();
        let samples = deinterleave_vecs_f32(&data.data, 1);

        let fft_size = 400;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16000.0;

        let mel_settings = MelConfig::new(fft_size, hop_size, n_mels, sampling_rate);
        let vad_settings = DetectionSettings::new(1.0, 10, 5, 0, 100);

        let config = PipelineConfig::new(mel_settings, Some(vad_settings));

        let mut pl = Pipeline::new(config);

        let handles = pl.start();

        // chunk size can be anything, 88 is random
        for chunk in samples[0].chunks(88) {
            let _ = pl.send_pcm(chunk);
        }

        pl.close_ingress();

        let mut res = Vec::new();

        while let Ok((idx, _)) = pl.rx().recv() {
            res.push(idx);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        //        assert_eq!(res.len(), 6);
    }
}
