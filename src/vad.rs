use image::{ImageBuffer, Rgb};
use ndarray::{concatenate, s, Array, Array2, Axis, Zip};

#[derive(Copy, Clone)]
pub struct DetectionSettings {
    pub energy_threshold: f64,
    pub min_intersections: usize,
    pub intersection_threshold: usize,
    pub min_mel: usize,
    pub min_frames: usize,
}

pub fn vad_boundaries(frames: &[Array2<f64>], settings: &DetectionSettings) -> EdgeInfo {
    let array_views: Vec<_> = frames.iter().map(|a| a.view()).collect();
    let threshold = settings.energy_threshold;
    let intersection_threshold = settings.intersection_threshold;
    let min_mel = settings.min_mel;
    // Concatenate the array views along Axis 0
    let merged_frames = concatenate(Axis(1), &array_views).unwrap();

    let shape = merged_frames.raw_dim();
    let width = shape[1];
    let height = shape[0];

    // Sobel kernels for gradient calculation
    let sobel_x =
        Array::from_shape_vec((3, 3), vec![-1., 0., 1., -2., 0., 2., -1., 0., 1.]).unwrap();
    let sobel_y =
        Array::from_shape_vec((3, 3), vec![-1., -2., -1., 0., 0., 0., 1., 2., 1.]).unwrap();

    // Convolve with Sobel kernels and calculate the gradient magnitude
    let gradient_mag = Array::from_shape_fn((height - 2, width - 2), |(y, x)| {
        let view = merged_frames.slice(s![y..y + 3, x..x + 3]);
        let mut gradient_x = 0.0;
        let mut gradient_y = 0.0;

        // Unroll the loop and compute gradient_x and gradient_y
        for j in 0..3 {
            for i in 0..3 {
                gradient_x += view[[j, i]].clone() * sobel_x[[j, i]];
                gradient_y += view[[j, i]].clone() * sobel_y[[j, i]];
            }
        }

        (gradient_x * gradient_x + gradient_y * gradient_y).sqrt()
    });

    // Count the number of high-energy gradients above the threshold
    let gradient_count = gradient_mag.iter().filter(|&val| *val >= threshold).count() as u32;

    let mut intersected_columns: Vec<usize> = Vec::new();
    let mut non_intersected_columns: Vec<usize> = Vec::new();

    for x in 0..width - 2 {
        let num_intersections = (min_mel..height - 2)
            .filter(|&y| gradient_mag[(y, x)] >= threshold)
            .count();
        if num_intersections < intersection_threshold {
            non_intersected_columns.push(x);
        }
        if num_intersections > intersection_threshold {
            intersected_columns.push(x);
        }
    }

    EdgeInfo {
        gradient_count,
        non_intersected_columns,
        intersected_columns,
    }
}

#[derive(Debug)]
pub struct EdgeInfo {
    gradient_count: u32,
    non_intersected_columns: Vec<usize>,
    intersected_columns: Vec<usize>,
}

impl EdgeInfo {
    pub fn non_intersected(&self) -> Vec<usize> {
        let val = self.non_intersected_columns.clone();
        val
    }
    pub fn intersected(&self) -> Vec<usize> {
        let val = self.intersected_columns.clone();
        val
    }
    pub fn gradient_count(&self) -> usize {
        let val = self.gradient_count.clone();
        val as usize
    }
}

pub fn as_image(
    frames: &[Array2<f64>],
    non_intersected_columns: &Vec<usize>,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let array_views: Vec<_> = frames.iter().map(|a| a.view()).collect();

    // Concatenate the array views along Axis 0
    let merged_frames = concatenate(Axis(1), &array_views).unwrap();

    let shape = merged_frames.raw_dim();
    let width = shape[1];
    let height = shape[0];
    let mut img_buffer = ImageBuffer::new(width as u32, height as u32);

    // Scale the values to the range [0, 255] for the grayscale image
    let max_val = merged_frames.fold(0.0, |acc: f64, &val| acc.max(val));
    let scaled_image: Array2<u8> = merged_frames.mapv(|val| (val * (255.0 / max_val)) as u8);

    for (y, row) in scaled_image.outer_iter().rev().enumerate() {
        for (x, &val) in row.into_iter().enumerate() {
            // Create an RGB pixel with all color channels set to the grayscale value
            let rgb_pixel = Rgb([val, val, val]);
            // Apply the green tint to the color channel for interesting columns
            if non_intersected_columns.contains(&x) {
                // Check if the current row is within the top 10 rows of the image
                if y < 10 {
                    // Set the pixel to be entirely green for the top 10 rows
                    let green_tint = Rgb([0, 255, 0]);
                    img_buffer.put_pixel(x as u32, y as u32, green_tint);
                } else {
                    // Apply a subtle green tint to the pixel for the rest of the rows
                    let green_tint_value = 60;
                    let green_tint = Rgb([val, val.saturating_add(green_tint_value), val]);
                    img_buffer.put_pixel(x as u32, y as u32, green_tint);
                }
            } else {
                // No tint for non-interesting columns, keep the grayscale value
                img_buffer.put_pixel(x as u32, y as u32, rgb_pixel);
            }
        }
    }

    img_buffer
}

/// An implementation of Voice-Activity-Detection for use in an audio pipeline
pub struct Stage {
    mel_buffer: Vec<Array2<f64>>,
    cutsec: usize,
    settings: DetectionSettings,
}

impl Stage {
    /// n_frames: the number of frames to accumulate before proceessing.
    ///           see n_frames_for_duration to get frames from milliseconds.  
    /// min_intersections: the min_intersections required in frames without intersecting gradients
    ///      before cutting. A value of 5 is recommended to detect word
    ///      boundaries in speech.
    pub fn new(settings: &DetectionSettings) -> Self {
        let mut mel_buffer: Vec<Array2<f64>> = Vec::new();
        let mut cutsec: usize = 0;

        Self {
            mel_buffer,
            cutsec,
            settings: settings.to_owned(),
        }
    }

    /// add Mel frame. Returns the frame start number and a full spectrogram
    /// once enough frames have accumulated.
    /// use duration_ms_for_n_frames to get the start time in milliseconds
    pub fn add(&mut self, frame: &Array2<f64>) -> Option<(usize, Vec<Array2<f64>>)> {
        self.mel_buffer.push(frame.to_owned());

        let buffer_len = self.mel_buffer.len();
        let min_intersections = self.settings.min_intersections;
        let min_frames = self.settings.min_frames;
        // i) check if have accumulated at least the min duration of spectrograms
        // ii) if no result, check again in steps of % n
        if buffer_len > min_frames {
            // check if we are at cutable frame position
            let window = &self.mel_buffer[buffer_len - min_intersections * 2..];
            let ni = vad_boundaries(&window, &self.settings).non_intersected();
            if ni.len() > 0 {
                if ni[ni.len() - 1] == ni.len() - 1 {
                    let idx = buffer_len - (min_intersections * 2);
                    let cutsec = self.cutsec.clone();
                    self.cutsec = self.cutsec.wrapping_add(idx);
                    // frames to process
                    let frames = &self.mel_buffer[..idx].to_vec();
                    // frames to carry forward to the new buffer
                    self.mel_buffer = self.mel_buffer[idx..].to_vec();

                    if frames.len() > 0 {
                        return Some((cutsec, frames.clone()));
                    }
                }
            }
        }
        None
    }

    /// Return the accumulated buffer in full, without doing edge detection.
    /// Useful to call at the end of a pipeline.
    pub fn flush(&self) -> Option<(usize, Vec<Array2<f64>>)> {
        Some((self.cutsec.clone(), self.mel_buffer.to_owned()))
    }
}

pub fn n_frames_for_duration(hop_size: usize, sampling_rate: f64, duration_ms: usize) -> usize {
    let frame_duration = hop_size as f32 / sampling_rate as f32 * 1000.0;
    let total_frames = (duration_ms as f32 / frame_duration).ceil() as u32;
    total_frames as usize
}

pub fn duration_ms_for_n_frames(hop_size: usize, sampling_rate: f64, total_frames: usize) -> usize {
    let frame_duration = hop_size as f64 / sampling_rate * 1000.0;
    (total_frames as f64 * frame_duration) as usize
}

pub fn format_milliseconds(milliseconds: u64) -> String {
    let total_seconds = milliseconds / 1000;
    let ms = milliseconds % 1000;
    let seconds = total_seconds % 60;
    let total_minutes = total_seconds / 60;
    let minutes = total_minutes % 60;
    let hours = total_minutes / 60;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, ms)
}

mod tests {
    use super::*;
    use crate::quant::{load_tga_8bit, to_array2};
    use std::time::Instant;

    #[test]
    fn test_stage() {
        let min_intersections = 4;
        let n_mels = 80;
        let settings = DetectionSettings {
            energy_threshold: 1.0,
            min_intersections: 4,
            intersection_threshold: 4,
            min_mel: 4,
            min_frames: 100,
        };
        let mut stage = Stage::new(&settings);

        let file_path = "./test/quantized_mel_golden.tga";
        let dequantized_mel = load_tga_8bit(file_path).unwrap();
        let frames = to_array2(&dequantized_mel, n_mels);
        let chunk_size = 1;
        let chunks: Vec<Array2<f64>> = frames
            .axis_chunks_iter(Axis(1), chunk_size)
            .map(|chunk| chunk.to_owned())
            .collect();

        let mut res = Vec::new();
        for mel in &chunks {
            if let Some((idx, _)) = stage.add(&mel) {
                res.push(idx);
            }
        }
    }
}
