use image::{ImageBuffer, Rgb};
use ndarray::{concatenate, s, Array, Array2, Axis};
use std::collections::HashSet;

#[derive(Copy, Clone, Default)]
pub struct DetectionSettings {
    pub min_energy: f64,
    pub min_y: usize,
    pub min_x: usize,
    pub min_mel: usize,
}

/// The purpose of these settings is to detect the "edges" of features in the
/// mel spectrogram, favouring gradients that are longer in the time axis and
/// above a certain power threshold.
///
/// Speech is characteristic for occupying several mel frequency bins at once
/// and continuing for several frames.
///
/// We naively sketch these waves as vectors, and find vertical columns where
/// there are no intersections above certain threshold - ie, short gaps in
/// speech.
///
/// `min_energy`: the relative power of the signal, set at around 1 to
///  discard noise.
/// `min_y`: this refers to the number of frames the gradient
///  interescts: i.e., its duration along the x-axis.
/// `min_x`: the number of distinct gradients that must
///  cross a column on the x-asis for it to be considered generally
///  intersected. The reasoning here is that speech will always occupy more
///  than one mel frequency bin so a time frame with speech will have several
///  or more intersections.
///  `min_mel`: bins below this wont be counted. Useful if the signal
///  is noisy in the very low (first couple of bins) frequencies.
///  `min_frames`: the min frames to accumulate before looking for a
///  boundary to split at. This should be at least 50-100. (100 frames is
///  1 second using Whisper FFT settings).
///
/// See `doc/jfk_vad_boundaries.png` for a visualisation.
impl DetectionSettings {
    pub fn new(min_energy: f64, min_y: usize, min_x: usize, min_mel: usize) -> Self {
        Self {
            min_energy,
            min_y,
            min_x,
            min_mel,
        }
    }

    /// Signals below this threshold will not be counted in edge detection.
    /// `1.0` is a good default.
    pub fn min_energy(&self) -> f64 {
        self.min_energy
    }

    /// The min length of a detectable gradient in x-asis frames.
    /// `10` is a good default.
    pub fn min_y(&self) -> usize {
        self.min_y
    }

    /// The min number of gradients allowed to intersect an x-axis before it is
    /// discounted from being a speech boundary.
    /// `10` is a good default.
    pub fn min_x(&self) -> usize {
        self.min_x
    }

    /// Ignore mel bands below this setting.
    /// `0` is a good default.
    pub fn min_mel(&self) -> usize {
        self.min_mel
    }
}

pub struct VoiceActivityDetector {
    mel_buffer: Vec<Array2<f64>>,
    settings: DetectionSettings,
    idx: usize,
}

impl VoiceActivityDetector {
    pub fn new(settings: &DetectionSettings) -> Self {
        let mel_buffer: Vec<Array2<f64>> = Vec::new();

        Self {
            mel_buffer,
            settings: settings.to_owned(),
            idx: 0,
        }
    }

    /// Add Mel spectrogram - should be a single frame.
    pub fn add(&mut self, frame: &Array2<f64>) -> Option<bool> {
        let min_x = self.settings.min_x;
        if self.idx == 128 {
            self.mel_buffer = self.mel_buffer[(self.mel_buffer.len() - min_x)..].to_vec();
            self.idx = min_x;
        }
        self.mel_buffer.push(frame.to_owned());
        self.idx += 1;
        if self.idx < min_x {
            return None;
        }

        // check if we are at cutable frame position
        let window = &self.mel_buffer[self.idx - min_x..];
        let edge_info = vad_boundaries(&window, &self.settings);
        let ni = edge_info.intersected();
        if ni.is_empty() {
            Some(false)
        } else {
            Some(ni[0] == 0)
        }
    }
}

fn vad_on(edge_info: &EdgeInfo, n: usize) -> bool {
    let intersected_columns = &edge_info.intersected_columns;

    if intersected_columns.is_empty() {
        return false; // No intersected columns, so return false
    }

    let mut contiguous_count = 1; // Count of contiguous intersected columns
    let mut prev_index = intersected_columns[0];

    for &index in &intersected_columns[1..] {
        if index == prev_index + 1 {
            contiguous_count += 1;
        } else {
            contiguous_count = 1;
        }

        if contiguous_count >= n {
            return true;
        }

        prev_index = index;
    }

    false // If no contiguous segment of n or more intersected columns is found, return false
}

/// Performs edge detection on the spectrogram using a fast Sobel operator
fn vad_boundaries(frames: &[Array2<f64>], settings: &DetectionSettings) -> EdgeInfo {
    let array_views: Vec<_> = frames.iter().map(|a| a.view()).collect();
    let min_energy = settings.min_energy;
    let min_y = settings.min_y;
    let min_mel = settings.min_mel;
    // Concatenate the array views along Axis 0
    let merged_frames = concatenate(Axis(1), &array_views).unwrap();
    let shape = merged_frames.raw_dim();
    let width = shape[1];
    let height = shape[0];

    // Sobel kernels for gradient calculation (to detect gradients along the X-axis)
    let sobel_x =
        Array::from_shape_vec((3, 3), vec![-1., 0., 1., -2., 0., 2., -1., 0., 1.]).unwrap();
    let sobel_y =
        Array::from_shape_vec((3, 3), vec![-1., -2., -1., 0., 0., 0., 1., 2., 1.]).unwrap();

    // Convolve with Sobel kernels and calculate the gradient magnitude along the Y-axis
    let gradient_mag = Array::from_shape_fn((height - 2, width - 2), |(y, x)| {
        if y < height && x < width {
            // Add boundary check to avoid going out of bounds
            let view = merged_frames.slice(s![y..y + 3, x..x + 3]);
            let mut gradient_x = 0.0;
            let mut gradient_y = 0.0;
            for j in 0..3 {
                for i in 0..3 {
                    gradient_x += view[[j, i]] * sobel_x[[j, i]]; // Use sobel_x for x-direction
                    gradient_y += view[[j, i]] * sobel_y[[j, i]]; // Use sobel_y for y-direction
                }
            }
            // Calculate the magnitude of the gradient (along Y-axis)
            (gradient_x * gradient_x + gradient_y * gradient_y).sqrt()
        } else {
            0.0
        }
    });

    let mut intersected_columns: Vec<usize> = Vec::new();
    let mut non_intersected_columns: Vec<usize> = Vec::new();
    let gradient_positions = HashSet::new();

    for x in 0..width - 2 {
        let indices: Vec<usize> = (0..height - 2)
            .filter(|&y| gradient_mag[(y, x)] >= min_energy && y >= min_mel)
            .collect();

        let num_intersections = indices.len();

        if num_intersections <= min_y {
            non_intersected_columns.push(x);
        } else if num_intersections >= min_y {
            intersected_columns.push(x);

            // Store the gradient positions for this column
            //            for y in indices {
            //                gradient_positions.insert((x, y));
            //:            }
        }
    }

    EdgeInfo::new(
        non_intersected_columns,
        intersected_columns,
        gradient_positions,
    )
}

/// EdgeInfo is the result of Voice Activity Detection.
/// `non_intersected_columns` are good places to cut and send to speech-to-text
#[derive(Debug)]
pub struct EdgeInfo {
    non_intersected_columns: Vec<usize>,
    intersected_columns: Vec<usize>,
    gradient_positions: HashSet<(usize, usize)>,
}

impl EdgeInfo {
    pub fn new(
        non_intersected_columns: Vec<usize>,
        intersected_columns: Vec<usize>,
        gradient_positions: HashSet<(usize, usize)>,
    ) -> Self {
        EdgeInfo {
            non_intersected_columns,
            intersected_columns,
            gradient_positions,
        }
    }

    /// The x-index of frames that don't intersect an edge.
    pub fn non_intersected(&self) -> Vec<usize> {
        self.non_intersected_columns.clone()
    }

    /// The x-index of frames that intersect an edge.
    pub fn intersected(&self) -> Vec<usize> {
        self.intersected_columns.clone()
    }

    ///  A bitmap, primarily used by [`as_image`].
    pub fn gradient_positions(&self) -> HashSet<(usize, usize)> {
        self.gradient_positions.clone()
    }
}

/// An image of the mel spectrogram, useful for testing detection settings.
/// Edge detection is overlayed in red and boundary detection in green.
pub fn as_image(
    frames: &[Array2<f64>],
    non_intersected_columns: &[usize],
    gradient_positions: &HashSet<(usize, usize)>,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let array_views: Vec<_> = frames.iter().map(|a| a.view()).collect();
    let array_view = concatenate(Axis(1), &array_views).unwrap();
    let shape = array_view.raw_dim();
    let width = shape[1];
    let height = shape[0];
    let mut img_buffer = ImageBuffer::new(width as u32, height as u32);

    let max_val = array_view.fold(0.0, |acc: f64, &val| acc.max(val));
    let scaled_image: Array2<u8> = array_view.mapv(|val| (val * (255.0 / max_val)) as u8);

    let tint_value = 200;

    for (y, row) in scaled_image.outer_iter().rev().enumerate() {
        for (x, &val) in row.into_iter().enumerate() {
            let mut rgb_pixel = Rgb([val, val, val]);

            if non_intersected_columns.contains(&x) {
                if y < 10 {
                    // Set the pixel to be entirely green for the top 10 rows
                    let green_tint = Rgb([0, 255, 0]);
                    rgb_pixel = green_tint;
                } else {
                    // Apply a subtle green tint to the pixel for the rest of the rows
                    let green_tint_value = 60;
                    let green_tint = Rgb([val, val.saturating_add(green_tint_value), val]);
                    rgb_pixel = green_tint;
                }
            }

            let inverted_y = height.checked_sub(y + 3).unwrap_or(0);
            if gradient_positions.contains(&(x, inverted_y)) {
                let tint = Rgb([tint_value, 0, 0]);
                rgb_pixel = Rgb([
                    rgb_pixel[0].saturating_add(tint[0]),
                    rgb_pixel[1].saturating_add(tint[1]),
                    rgb_pixel[2].saturating_add(tint[2]),
                ]);
            }

            img_buffer.put_pixel(x as u32, y as u32, rgb_pixel);
        }
    }

    img_buffer
}

/// Returns number of FFT frames are needed for nth milliseconds
pub fn n_frames_for_duration(hop_size: usize, sampling_rate: f64, duration_ms: usize) -> usize {
    let frame_duration = hop_size as f32 / sampling_rate as f32 * 1000.0;
    let total_frames = (duration_ms as f32 / frame_duration).ceil() as u32;
    total_frames as usize
}

/// Returns milliseconds nth FFT frames represent
pub fn duration_ms_for_n_frames(hop_size: usize, sampling_rate: f64, total_frames: usize) -> usize {
    let frame_duration = hop_size as f64 / sampling_rate * 1000.0;
    (total_frames as f64 * frame_duration) as usize
}

/// Formats milliseconds to HH:MM:SS.MS
pub fn format_milliseconds(milliseconds: u64) -> String {
    let total_seconds = milliseconds / 1000;
    let ms = milliseconds % 1000;
    let seconds = total_seconds % 60;
    let total_minutes = total_seconds / 60;
    let minutes = total_minutes % 60;
    let hours = total_minutes / 60;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, ms)
}

/// Smoke test - see the generated `./test/vad.png`.
/// green lines are the cutsecs predicted to not intersect speech,
/// red pixels are the detected gradients.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{load_tga_8bit, to_array2};

    #[test]
    fn test_speech_detection() {
        let n_mels = 80;
        let min_x = 5;
        let settings = DetectionSettings {
            min_energy: 1.0,
            min_y: 10,
            min_x,
            min_mel: 0,
        };

        let ids = vec![21168, 23760, 41492, 41902, 63655, 7497, 39744];
        for id in ids {
            let file_path = format!("./testdata/blank/frame_{}.tga", id);
            let dequantized_mel = load_tga_8bit(&file_path).unwrap();
            let frames = to_array2(&dequantized_mel, n_mels);

            let edge_info = vad_boundaries(&[frames.clone()], &settings);
            let img = as_image(
                &[frames.clone()],
                &edge_info.non_intersected(),
                &edge_info.gradient_positions(),
            );

            assert!(vad_on(&edge_info, min_x) == false);
            let path = format!("./testdata/vad_off_{}.png", id);
            img.save(path).unwrap();
        }

        let ids = vec![11648, 2889, 4694, 4901, 27125];
        for id in ids {
            let file_path = format!("./testdata/speech/frame_{}.tga", id);
            let dequantized_mel = load_tga_8bit(&file_path).unwrap();
            let frames = to_array2(&dequantized_mel, n_mels);

            let edge_info = vad_boundaries(&[frames.clone()], &settings);
            let img = as_image(
                &[frames.clone()],
                &edge_info.non_intersected(),
                &edge_info.gradient_positions(),
            );

            assert!(vad_on(&edge_info, min_x) == true);
            let path = format!("./testdata/vad_on_{}.png", id);
            img.save(path).unwrap();

            //assert!(edge_info.gradient_count > 800);
        }
    }

    //#[test]
    fn test_vad_debug() {
        let n_mels = 80;
        let settings = DetectionSettings {
            min_energy: 1.0,
            min_y: 6,
            min_x: 1,
            min_mel: 0,
        };

        let start = std::time::Instant::now();
        let file_path = "./testdata/jfk_full_speech_chunk0_golden.tga";
        let dequantized_mel = load_tga_8bit(file_path).unwrap();
        let frames = to_array2(&dequantized_mel, n_mels);

        let edge_info = vad_boundaries(&[frames.clone()], &settings);

        let elapsed = start.elapsed().as_millis();
        dbg!(elapsed);
        let img = as_image(
            &[frames.clone()],
            &edge_info.non_intersected(),
            &edge_info.gradient_positions(),
        );

        img.save("./doc/debug.png").unwrap();
    }

    #[test]
    fn test_vad_boundaries() {
        let n_mels = 80;
        let settings = DetectionSettings {
            min_energy: 1.0,
            min_y: 3,
            min_x: 6,
            min_mel: 0,
        };

        let start = std::time::Instant::now();
        let file_path = "./testdata/quantized_mel_golden.tga";
        let dequantized_mel = load_tga_8bit(file_path).unwrap();
        dbg!(&dequantized_mel);

        let frames = to_array2(&dequantized_mel, n_mels);

        let edge_info = vad_boundaries(&[frames.clone()], &settings);

        let elapsed = start.elapsed().as_millis();
        dbg!(elapsed);
        let img = as_image(
            &[frames.clone()],
            &edge_info.non_intersected(),
            &edge_info.gradient_positions(),
        );

        img.save("./doc/vad.png").unwrap();
    }

    //    #[test]
    fn test_stage() {
        let n_mels = 80;
        let settings = DetectionSettings {
            min_energy: 1.0,
            min_y: 3,
            min_x: 3,
            min_mel: 0,
        };
        let mut stage = VoiceActivityDetector::new(&settings);

        let file_path = "./testdata/quantized_mel_golden.tga";
        let dequantized_mel = load_tga_8bit(file_path).unwrap();
        let frames = to_array2(&dequantized_mel, n_mels);
        let chunk_size = 1;
        let chunks: Vec<Array2<f64>> = frames
            .axis_chunks_iter(Axis(1), chunk_size)
            .map(|chunk| chunk.to_owned())
            .collect();

        let start = std::time::Instant::now();

        for mel in &chunks {
            if let Some(_) = stage.add(&mel) {}
        }
        let elapsed = start.elapsed().as_millis();
        dbg!(elapsed);
    }
}
