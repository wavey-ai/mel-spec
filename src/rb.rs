use crate::{config::MelConfig, mel::MelSpectrogram, stft};
use ndarray::Array2;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub struct RingBuffer {
    accumulated_samples: Vec<f32>,
    buffer: VecDeque<f32>,
    fft: stft::Spectrogram,
    mel: MelSpectrogram,
    config: MelConfig,
}

impl RingBuffer {
    pub fn new(config: MelConfig, capacity: usize) -> Self {
        let hop_size = config.hop_size();
        let fft_size = config.fft_size();
        let sample_rate = config.sampling_rate();

        let buffer = VecDeque::with_capacity(capacity);
        let accumulated_samples = Vec::with_capacity(hop_size);
        let mel = MelSpectrogram::new(fft_size, sample_rate, config.n_mels());
        let fft = stft::Spectrogram::new(fft_size, hop_size);

        Self {
            config,
            buffer,
            accumulated_samples,
            mel,
            fft,
        }
    }

    pub fn add_frame(&mut self, samples: &[f32]) {
        let available_capacity = self.buffer.capacity() - self.buffer.len();
        if samples.len() > available_capacity {
            // If incoming samples exceed available capacity, remove the excess from the front
            self.buffer.drain(0..samples.len() - available_capacity);
        }
        self.buffer.extend(samples);
    }

    pub fn add(&mut self, sample: f32) {
        if self.buffer.len() == self.buffer.capacity() {
            self.buffer.pop_front();
        }
        self.buffer.push_back(sample);
    }

    pub fn maybe_mel(&mut self) -> Option<Array2<f64>> {
        let hop_size = self.config.hop_size();

        if self.accumulated_samples.len() >= hop_size {
            // Accumulated samples are sufficient to process
            let frame = self.accumulated_samples.split_off(hop_size);
            std::mem::swap(&mut self.accumulated_samples, &mut frame.clone());
        } else {
            // Accumulate more samples from the buffer
            let to_add = hop_size - self.accumulated_samples.len();
            let available = self.buffer.len().min(to_add); // Ensure we don't try to drain more than available
            self.accumulated_samples
                .extend(self.buffer.drain(..available));
        }

        if self.accumulated_samples.len() < hop_size {
            return None;
        }

        let fft_result = self.fft.add(&self.accumulated_samples);
        self.accumulated_samples.clear();

        fft_result.map(|fft| self.mel.add(&fft))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::interleave_frames;
    use crate::quant::save_tga_8bit;
    use ndarray_npy::write_npy;
    use regex::Regex;
    use soundkit::{
        audio_bytes::{deinterleave_vecs_f32, deinterleave_vecs_i16},
        wav::WavStreamProcessor,
    };
    use std::fs;
    use std::fs::File;
    use std::io::Read;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    #[test]
    fn test_load_tensor_to_tga() {
        use crate::mel::interleave_frames;
        use crate::quant::save_tga_8bit;
        use ndarray::{Array, Array2};
        use ndarray_npy::{read_npy, write_npy};
        use std::fs;
        use std::path::Path;

        // Path to the exported tensor
        let tensor_path = "./testdata/exported_audio.npy";
        let output_tga_path = "./testdata/exported_spectrogram.tga";

        println!("Looking for tensor file at: {}", tensor_path);

        // Check if the file exists first
        if !Path::new(tensor_path).exists() {
            panic!("File not found: {}. Please ensure the Python code has exported the tensor to this location.", tensor_path);
        }

        // Since we can't directly load the combined data as a HashMap in ndarray-npy,
        // we'll load the features array directly
        let features: Array2<f32> = match read_npy(Path::new(tensor_path)) {
            Ok(array) => {
                println!("Successfully loaded NPY file");
                array
            }
            Err(e) => {
                // Provide detailed error information
                panic!("Failed to read NPY file at {}: {}. Try exporting a simple tensor directly with np.save(path, tensor.cpu().numpy()).", 
                   tensor_path, e);
            }
        };

        println!("Loaded features with shape: {:?}", features.shape());

        println!("Loaded features with shape: {:?}", features.shape());

        // Convert features to f64 (as your existing code uses f64 for mel spectrograms)
        let features_f64 = features.mapv(|x| x as f64);

        // Prepare for TGA conversion
        // Assuming features is already in the mel spectrogram format
        // We need to convert it to the format expected by save_tga_8bit

        // Based on your existing code, we need to prepare frames
        // Check if our tensor is already in the right format (n_mels x time_steps)
        // or if we need to reshape it

        let (n_mels, time_steps) = features_f64.dim();
        println!("Dimensions: {} mels x {} time steps", n_mels, time_steps);

        // If we need to convert to a vector of Array2 frames:
        let mut frames: Vec<Array2<f64>> = Vec::new();
        for t in 0..time_steps {
            // For each time step, extract the mel spectrogram frame
            let frame = features_f64
                .slice(ndarray::s![.., t])
                .to_owned()
                .into_shape((n_mels, 1))
                .expect("Failed to reshape frame");
            frames.push(frame);
        }

        // Interleave the frames as done in the existing code
        let flattened_frames = interleave_frames(&frames, false, 0);

        // Save as TGA
        match save_tga_8bit(&flattened_frames, n_mels, output_tga_path) {
            Ok(_) => println!("Successfully saved TGA to: {}", output_tga_path),
            Err(e) => panic!("Error saving TGA file: {}", e),
        }

        // Optional: Also save the original tensor as NPY for verification
        let stacked_frames = Array2::from_shape_vec((n_mels, time_steps), flattened_frames)
            .expect("Error reshaping flattened frames");

        let verification_path = "./testdata/verification.npy";
        write_npy(verification_path, &stacked_frames).expect("Failed to write verification NPY");

        println!("Saved verification NPY to: {}", verification_path);
    }

    //    #[test]
    fn test_harvard_wavs_to_tga() {
        let fft_size = 512;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16000.0;

        let config = MelConfig::new(fft_size, hop_size, n_mels, sampling_rate);

        // Path to the harvard directory containing WAV files
        let harvard_dir = Path::new("/Users/jamieb/wavey.ai/harvard-lines/output2/");

        // Output directory for TGA files
        let output_dir = Path::new("./harvard");

        // Create output directory if it doesn't exist
        if !output_dir.exists() {
            fs::create_dir_all(output_dir).expect("Failed to create output directory");
        }

        let wav_files = find_wav_files(harvard_dir);

        if wav_files.is_empty() {
            panic!("No WAV files found in the harvard directory");
        }

        println!("Found {} WAV files to process", wav_files.len());

        // Process each WAV file
        for wav_path in wav_files {
            process_wav_to_tga(&wav_path, output_dir, &config);
        }
    }

    // Helper function to recursively find all WAV files in a directory
    fn find_wav_files(dir: &Path) -> Vec<PathBuf> {
        let mut wav_files = Vec::new();

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();

                if path.is_dir() {
                    // Recursive call for subdirectories
                    wav_files.extend(find_wav_files(&path));
                } else if let Some(extension) = path.extension() {
                    // Check if the file has a .wav extension
                    if extension.to_string_lossy().to_lowercase() == "wav" {
                        wav_files.push(path);
                    }
                }
            }
        }

        wav_files
    }

    // Extract number from Harvard list filename
    fn extract_harvard_number(filename: &str) -> Option<u32> {
        // Create regex to match "Harvard list XX" or similar patterns
        let re = Regex::new(r"(?i)harvard\s*(?:list)?\s*(\d+)").unwrap();

        if let Some(captures) = re.captures(filename) {
            if let Some(num_match) = captures.get(1) {
                return num_match.as_str().parse::<u32>().ok();
            }
        }

        None
    }

    // Process a single WAV file to a TGA file
    fn process_wav_to_tga(wav_path: &Path, output_dir: &Path, config: &MelConfig) {
        println!("Processing: {:?}", wav_path);

        // Extract filename from path
        let filename = wav_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");

        // Extract number from Harvard filename
        let file_number = extract_harvard_number(filename).unwrap_or_else(|| {
            println!(
                "Warning: Could not extract Harvard number from {:?}, using default",
                filename
            );
            0 // Default if we can't extract a number
        });

        // Create new filename in format "H{n}.tga"
        let tga_filename = format!("H{}.tga", file_number);
        let tga_path = output_dir.join(tga_filename);

        let mut file = match File::open(wav_path) {
            Ok(f) => f,
            Err(e) => {
                println!("Error opening file {:?}: {}", wav_path, e);
                return;
            }
        };

        let mut processor = WavStreamProcessor::new();
        let mut rb = RingBuffer::new(config.clone(), 1024);
        let mut buffer = [0u8; 1024]; // Larger buffer for faster processing
        let mut frames: Vec<Array2<f64>> = Vec::new();

        // Start timing the mel tensor creation process
        let mel_start_time = Instant::now();

        loop {
            let bytes_read = match file.read(&mut buffer) {
                Ok(0) => break, // End of file
                Ok(n) => n,
                Err(e) => {
                    println!("Error reading file {:?}: {}", wav_path, e);
                    return;
                }
            };

            let chunk = &buffer[..bytes_read];
            match processor.add(chunk) {
                Ok(Some(audio_data)) => {
                    // The audio data is expected to be mono (1-channel)
                    let i16_samples = deinterleave_vecs_i16(audio_data.data(), 1);
                    let samples: Vec<f32> =
                        i16_samples[0].iter().map(|&s| s as f32 / 32768.0).collect();
                    rb.add_frame(&samples);

                    // Process any available frames
                    while let Some(mel_frame) = rb.maybe_mel() {
                        frames.push(mel_frame);
                    }
                }
                Ok(None) => continue, // Need more data
                Err(err) => {
                    println!("Error processing audio data from {:?}: {}", wav_path, err);
                    return;
                }
            }
        }

        // Process any remaining frames
        while let Some(mel_frame) = rb.maybe_mel() {
            frames.push(mel_frame);
        }

        // Stop timing the mel tensor creation process
        let mel_processing_time = mel_start_time.elapsed();
        println!(
            "Mel tensor creation time for {:?}: {} ms",
            filename,
            mel_processing_time.as_millis()
        );

        if frames.is_empty() {
            println!("Warning: No frames were generated for {:?}", wav_path);
            return;
        }

        // Start timing the TGA saving process (separate from mel creation)
        let tga_start_time = Instant::now();

        // Interleave the frames in row-major order (the format expected by TGA)
        let flattened_frames = interleave_frames(&frames, false, 0);

        // Save as TGA
        match save_tga_8bit(
            &flattened_frames,
            config.n_mels(),
            tga_path.to_str().unwrap(),
        ) {
            Ok(_) => {
                let tga_save_time = tga_start_time.elapsed();
                println!(
                    "Successfully saved: {:?} (TGA save time: {} ms)",
                    tga_path,
                    tga_save_time.as_millis()
                );
            }
            Err(e) => println!("Error saving TGA file {:?}: {}", tga_path, e),
        }
    }

    //    #[test]
    fn test_ringbuffer() {
        let fft_size = 512;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16000.0;

        let config = MelConfig::new(fft_size, hop_size, n_mels, sampling_rate);

        let mut rb = RingBuffer::new(config, 1024);

        // 16000 Hz, mono, flt
        let file_path = "./testdata/jfk_f32le.wav";
        let mut file = File::open(&file_path).unwrap();

        let mut processor = WavStreamProcessor::new();
        let mut buffer = [0u8; 128];

        let mut frames: Vec<Array2<f64>> = Vec::new();
        loop {
            let bytes_read = file.read(&mut buffer).unwrap();
            if bytes_read == 0 {
                break;
            }

            let chunk = &buffer[..bytes_read];
            match processor.add(chunk) {
                Ok(Some(audio_data)) => {
                    // the test data is f32 mono
                    let samples = deinterleave_vecs_f32(audio_data.data(), 1);
                    rb.add_frame(&samples[0]);
                    if let Some(mel_frame) = rb.maybe_mel() {
                        frames.push(mel_frame);
                    }
                }
                Ok(None) => continue,
                Err(err) => panic!("Error: {}", err),
            }
        }

        let flattened_frames = interleave_frames(&frames, false, 0);

        let num_time_steps = frames.len();
        let num_frequency_bands = frames[0].dim().0; // Assuming all frames have the same number of rows

        // Reshape the flattened frames into a 2D array
        let stacked_frames =
            Array2::from_shape_vec((num_frequency_bands, num_time_steps), flattened_frames)
                .expect("Error reshaping flattened frames");

        let _ = write_npy("./testdata/rust_jfk.npy", &stacked_frames);
    }
}
