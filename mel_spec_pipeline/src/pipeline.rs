use crossbeam_channel::{bounded, unbounded, Receiver, SendError, Sender};
use mel_spec::prelude::*;
use mel_spec::vad::duration_ms_for_n_frames;
use ndarray::{Array1, Array2};
use num::Complex;
use rubato::{FastFixedIn, PolynomialDegree, Resampler};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug)]
pub struct MelFrame {
    mel: Array2<f64>,
    idx: usize,
}

impl MelFrame {
    pub fn new(mel: Array2<f64>, idx: usize) -> Self {
        MelFrame { mel, idx }
    }

    pub fn frame(&self) -> &Array2<f64> {
        &self.mel
    }

    pub fn idx(&self) -> usize {
        self.idx
    }
}

#[derive(Debug)]
pub struct VadResult {
    start: usize,
    end: usize,
    ms: usize,
}

impl VadResult {
    pub fn new(start: usize, end: usize, ms: usize, active: bool, gaps: Vec<usize>) -> Self {
        VadResult { start, end, ms }
    }

    pub fn get_start(&self) -> usize {
        self.start
    }

    pub fn get_end(&self) -> usize {
        self.end
    }

    pub fn get_ms(&self) -> usize {
        self.ms
    }
}

pub struct AudioConfig {
    bit_depth: usize,
    sampling_rate: f64,
}

impl AudioConfig {
    pub fn new(bit_depth: usize, sampling_rate: f64) -> Self {
        Self {
            bit_depth,
            sampling_rate,
        }
    }
}

pub struct PipelineConfig {
    vad_config: DetectionSettings,
    mel_config: MelConfig,
    audio_config: AudioConfig,
}

impl PipelineConfig {
    pub fn new(
        audio_config: AudioConfig,
        mel_config: MelConfig,
        vad_config: DetectionSettings,
    ) -> Self {
        Self {
            audio_config,
            mel_config,
            vad_config,
        }
    }
}
pub struct Pipeline {
    // we drop pcm_tx by setting it to None, thereby removing all references to it.
    pcm_tx: Option<Sender<Vec<f32>>>,
    pcm_rx: Receiver<Vec<f32>>,
    mel_tx: Arc<Mutex<Option<Sender<MelFrame>>>>,
    mel_rx: Receiver<MelFrame>,
    vad_tx: Arc<Mutex<Option<Sender<VadResult>>>>,
    vad_rx: Receiver<VadResult>,
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
        let (pcm_tx, pcm_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();
        let (mel_tx, mel_rx): (Sender<MelFrame>, Receiver<MelFrame>) = unbounded();
        let (vad_tx, vad_rx): (Sender<VadResult>, Receiver<VadResult>) = unbounded();

        let mel_tx_arc = Arc::new(Mutex::new(Some(mel_tx)));
        let vad_tx_arc = Arc::new(Mutex::new(Some(vad_tx)));

        Self {
            pcm_tx: Some(pcm_tx),
            pcm_rx,
            mel_tx: mel_tx_arc,
            mel_rx,
            vad_tx: vad_tx_arc,
            vad_rx,
            config,
        }
    }

    /// send PCM mono, non-interleaved f32 samples - chunks can be any length.
    pub fn send_pcm(&self, pcm: &[f32]) -> Result<(), SendError<Vec<f32>>> {
        self.pcm_tx.as_ref().expect("not closed").send(pcm.to_vec())
    }

    /// receive spectrogram
    pub fn mel_rx(&self) -> Receiver<MelFrame> {
        self.mel_rx.clone()
    }

    /// receive voice-activity information
    pub fn vad_rx(&self) -> Receiver<VadResult> {
        self.vad_rx.clone()
    }

    /// to signal we are done streaming into the pipeline
    pub fn close_ingress(&mut self) {
        self.pcm_tx = None;
    }

    /// start background threads
    pub fn start(&mut self) -> Vec<thread::JoinHandle<()>> {
        // avoid cloning self
        let fft_size = self.config.mel_config.fft_size();
        let hop_size = self.config.mel_config.hop_size();
        let n_mels = self.config.mel_config.n_mels();
        let sampling_rate = self.config.mel_config.sampling_rate();
        let source_sampling_rate = self.config.audio_config.sampling_rate;
        let vad_settings = self.config.vad_config;
        let mut handles = Vec::new();

        let (samples_tx, samples_rx): (Sender<(usize, Vec<f32>)>, Receiver<(usize, Vec<f32>)>) =
            unbounded();

        let (sr_tx, sr_rx): (Sender<(usize, Vec<f32>)>, Receiver<(usize, Vec<f32>)>) = unbounded();

        let (mel_stt_tx, mel_stt_rx): (
            Sender<(usize, Array1<Complex<f64>>)>,
            Receiver<(usize, Array1<Complex<f64>>)>,
        ) = bounded(1);

        let (mel_vad_tx, mel_vad_rx): (
            Sender<(usize, Array1<Complex<f64>>)>,
            Receiver<(usize, Array1<Complex<f64>>)>,
        ) = bounded(1);

        let (mels_tx, mels_rx): (Sender<(usize, Array2<f64>)>, Receiver<(usize, Array2<f64>)>) =
            bounded(2);

        let pcm_rx_clone = self.pcm_rx.clone();
        let mel_tx_clone = Arc::clone(&self.mel_tx);
        let vad_tx_clone = Arc::clone(&self.vad_tx);

        let mels_tx_clone = mels_tx.clone();
        let mel_handle = thread::spawn(move || {
            let mut mel = MelSpectrogram::new(fft_size, sampling_rate, n_mels);

            while let Ok((idx, fft)) = mel_stt_rx.recv() {
                let spec = mel.add(&fft);
                if let Err(send_error) = mels_tx.send((idx, spec)) {}
            }

            drop(mels_tx);
        });

        handles.push(mel_handle);

        let mel_vad_handle = thread::spawn(move || {
            let mut mel = MelSpectrogram::new(fft_size, sampling_rate, n_mels / 4);

            while let Ok((idx, fft)) = mel_vad_rx.recv() {
                let spec = mel.add(&fft);
                if let Err(send_error) = mels_tx_clone.send((idx, spec)) {}
            }

            drop(mels_tx_clone);
        });

        handles.push(mel_vad_handle);

        let vad_handle = thread::spawn(move || {
            let mut vad = VoiceActivityDetector::new(&vad_settings);
            let mut i = 0;
            let mut mels: Vec<Array2<f64>> = vec![
                Array2::from_elem((0, 0), 0.0),
                Array2::from_elem((0, 0), 0.0),
            ];
            let mut last_cutsec = 0;
            while let Ok((idx, fft)) = mels_rx.recv() {
                if i == 2 {
                    for (j, m) in mels.iter().enumerate() {
                        if m.raw_dim()[0] != n_mels {
                            let mel = mels[if j == 1 { 0 } else { 1 }].clone();
                            if let Some(gap) = vad.add(m) {
                                let result = VadResult {
                                    start: last_cutsec,
                                    end: idx,
                                    ms: duration_ms_for_n_frames(hop_size, sampling_rate, idx),
                                };
                                if let Some(tx) = vad_tx_clone.lock().unwrap().as_ref() {
                                    if let Err(send_error) = tx.send(result) {}
                                }
                                last_cutsec = idx.clone();
                            } else {
                                let result = MelFrame { idx, mel };
                                if let Some(tx) = mel_tx_clone.lock().unwrap().as_ref() {
                                    if let Err(send_error) = tx.send(result) {}
                                }
                            }
                        }

                        i = 0;
                    }
                }
                mels[i] = fft;
                i += 1;
            }

            if let Some(_) = mel_tx_clone.lock().unwrap().take() {
                // At this point, mel_tx goes out of scope and will be dropped
                // The lock is automatically released when mel_tx goes out of scope
            }

            if let Some(_) = vad_tx_clone.lock().unwrap().take() {
                // At this point, vad_tx goes out of scope and will be dropped
                // The lock is automatically released when vad_tx goes out of scope
            }
        });

        handles.push(vad_handle);

        let fft_handle = thread::spawn(move || {
            let mut fft = Spectrogram::new(fft_size, hop_size);

            while let Ok((idx, samples)) = samples_rx.recv() {
                // buffer up to an initial fft window
                if let Some(complex) = fft.add(&samples) {
                    if let Err(send_error) = mel_stt_tx.send((idx, complex.clone())) {}
                    if let Err(send_error) = mel_vad_tx.send((idx, complex)) {}
                }
            }

            drop(mel_stt_tx);
            drop(mel_vad_tx);
        });

        handles.push(fft_handle);

        let sr_handle = thread::spawn(move || {
            let f_ratio = sampling_rate / source_sampling_rate;
            let target_ratio = source_sampling_rate / sampling_rate;
            let mut resampler = FastFixedIn::<f32>::new(
                f_ratio,
                target_ratio,
                PolynomialDegree::Cubic,
                hop_size,
                1,
            )
            .unwrap();
            while let Ok((idx, samples)) = sr_rx.recv() {
                let samples = resampler.process(&[samples], None).unwrap();
                let chan = samples[0].clone();
                if let Err(send_error) = samples_tx.send((idx, chan)) {
                    eprintln!("Failed to send message to fft: {:?}", send_error);
                }
            }

            drop(samples_tx);
        });

        handles.push(sr_handle);

        // handler for external audio source.
        let pcm_handle = thread::spawn(move || {
            let mut idx = 0;
            let mut accumulated_samples: Vec<f32> = Vec::new();
            while let Ok(samples) = pcm_rx_clone.recv() {
                accumulated_samples.extend_from_slice(&samples);
                while accumulated_samples.len() >= hop_size {
                    let (chunk, rest) = accumulated_samples.split_at(hop_size);
                    sr_tx.send((idx.clone(), chunk.to_vec())).unwrap();
                    idx = idx.wrapping_add(1);
                    accumulated_samples = rest.to_vec();
                }
            }

            if !accumulated_samples.is_empty() {
                idx = idx.wrapping_add(1);
                sr_tx
                    .send((idx.clone(), accumulated_samples.clone()))
                    .unwrap();
            }

            drop(sr_tx);
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

    pub fn add(&mut self, idx: usize, frame: &Array2<f64>) -> Option<Vec<f32>> {
        if idx != self.idx {
            let window = self.buffer.clone();
            self.buffer.drain(..);
            self.idx = idx;
            let frames: Vec<f32> = interleave_frames(&window, false, 100);
            return Some(frames);
        } else {
            self.buffer.push(frame.to_owned());
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

        let audio_config = AudioConfig::new(32, 16000.0);
        let mel_config = MelConfig::new(fft_size, hop_size, n_mels, sampling_rate);
        let vad_config = DetectionSettings::new(1.0, 3, 6, 0);

        let config = PipelineConfig::new(audio_config, mel_config, vad_config);

        let mut pl = Pipeline::new(config);

        let start = std::time::Instant::now();
        let handles = pl.start();

        // chunk size can be anything, 88 is random
        for chunk in samples[0].chunks(88) {
            let _ = pl.send_pcm(chunk);
        }

        pl.close_ingress();

        while let Ok(res) = pl.vad_rx().recv() {
            //dbg!(res);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let elapsed = start.elapsed().as_millis();
        dbg!(elapsed);

        //        assert_eq!(res.len(), 6);
    }
}
