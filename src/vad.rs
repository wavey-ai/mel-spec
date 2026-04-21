use image::{ImageBuffer, Rgb};
use ndarray::{concatenate, Array2, Axis};
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
}

impl VoiceActivityDetector {
    pub fn new(settings: &DetectionSettings) -> Self {
        let mel_buffer: Vec<Array2<f64>> = Vec::new();

        Self {
            mel_buffer,
            settings: settings.to_owned(),
        }
    }

    /// Add Mel spectrogram - should be a single frame.
    pub fn add(&mut self, frame: &Array2<f64>) -> Option<bool> {
        let min_x = self.settings.min_x;
        self.mel_buffer.push(frame.to_owned());
        let max_buffered_frames = min_x.max(128);
        if self.mel_buffer.len() > max_buffered_frames {
            let keep_from = self.mel_buffer.len().saturating_sub(min_x);
            self.mel_buffer.drain(..keep_from);
        }
        if self.mel_buffer.len() < min_x {
            return None;
        }

        // check if we are at cutable frame position
        let window = &self.mel_buffer[self.mel_buffer.len() - min_x..];
        let edge_info = vad_boundaries(&window, &self.settings);
        let intersected = &edge_info.intersected_columns;
        if intersected.is_empty() {
            Some(false)
        } else {
            Some(intersected[0] == 0)
        }
    }
}

pub fn vad_on(edge_info: &EdgeInfo, n: usize) -> bool {
    let intersected_columns = &edge_info.intersected_columns;

    if intersected_columns.is_empty() {
        return false;
    }

    let mut contiguous_count = 1;
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

    false
}

pub fn vad_boundaries(frames: &[Array2<f64>], settings: &DetectionSettings) -> EdgeInfo {
    let Some(first_frame) = frames.first() else {
        return EdgeInfo::new(Vec::new(), Vec::new(), HashSet::new());
    };

    let height = first_frame.nrows();
    let width: usize = frames
        .iter()
        .map(|frame| {
            debug_assert_eq!(frame.nrows(), height);
            frame.ncols()
        })
        .sum();

    if height < 3 || width < 3 {
        return EdgeInfo::new(Vec::new(), Vec::new(), HashSet::new());
    }

    let mut raw_classification = vec![false; width - 2];
    let min_energy_sq = settings.min_energy * settings.min_energy;

    if frames.len() == 1 {
        let frame = &frames[0];
        let data = frame
            .as_slice()
            .expect("VAD expects contiguous spectrogram frames");
        classify_columns_in_frame(
            data,
            frame.ncols(),
            height,
            settings.min_mel,
            settings.min_y,
            min_energy_sq,
            &mut raw_classification,
        );
    } else {
        let mut frame_infos = Vec::with_capacity(frames.len());
        let mut column_sources = Vec::with_capacity(width);

        for frame in frames {
            let frame_index = frame_infos.len();
            frame_infos.push(FrameInfo {
                data: frame
                    .as_slice()
                    .expect("VAD expects contiguous spectrogram frames"),
                width: frame.ncols(),
            });
            column_sources.extend((0..frame.ncols()).map(|local_x| ColumnSource {
                frame_index,
                local_x,
            }));
        }

        classify_columns_across_frames(
            &frame_infos,
            &column_sources,
            height,
            settings.min_mel,
            settings.min_y,
            min_energy_sq,
            &mut raw_classification,
        );
    }

    // Apply temporal smoothing via a moving-window majority vote.
    // For each index, we consider a window of neighboring columns (window size can be adjusted).
    let smoothed_classification = smooth_mask(&raw_classification, 4);

    // Split the smoothed results into active (intersected) and inactive (non-intersected) columns.
    let mut intersected_columns = Vec::new();
    let mut non_intersected_columns = Vec::new();
    for (x, &active) in smoothed_classification.iter().enumerate() {
        if active {
            intersected_columns.push(x);
        } else {
            non_intersected_columns.push(x);
        }
    }

    // We leave gradient_positions empty in this version.
    let gradient_positions = HashSet::new();

    EdgeInfo::new(
        non_intersected_columns,
        intersected_columns,
        gradient_positions,
    )
}

/// Applies a simple temporal smoothing (moving-window majority vote) over a binary mask.
/// For each index, we look at the window of values [i-window, i+window] and set the smoothed
/// value to true if at least half of the values in that window are true.
fn smooth_mask(mask: &[bool], window: usize) -> Vec<bool> {
    let n = mask.len();
    let mut prefix_true = vec![0usize; n + 1];
    for (i, &value) in mask.iter().enumerate() {
        prefix_true[i + 1] = prefix_true[i] + usize::from(value);
    }

    let mut smoothed = vec![false; n];
    for i in 0..n {
        let start = i.saturating_sub(window);
        let end = (i + window + 1).min(n);
        let count_true = prefix_true[end] - prefix_true[start];
        if count_true * 2 >= (end - start) {
            smoothed[i] = true;
        }
    }
    smoothed
}

struct FrameInfo<'a> {
    data: &'a [f64],
    width: usize,
}

#[derive(Copy, Clone)]
struct ColumnSource {
    frame_index: usize,
    local_x: usize,
}

fn classify_columns_in_frame(
    frame: &[f64],
    width: usize,
    height: usize,
    min_mel: usize,
    min_y: usize,
    min_energy_sq: f64,
    output: &mut [bool],
) {
    if min_y == 0 {
        output.fill(true);
        return;
    }

    let start_y = min_mel.min(height - 2);

    for (x, is_active) in output.iter_mut().enumerate() {
        let mut count = 0;

        for y in start_y..(height - 2) {
            let row0 = y * width;
            let row1 = row0 + width;
            let row2 = row1 + width;

            let tl = frame[row0 + x];
            let tc = frame[row0 + x + 1];
            let tr = frame[row0 + x + 2];
            let ml = frame[row1 + x];
            let mr = frame[row1 + x + 2];
            let bl = frame[row2 + x];
            let bc = frame[row2 + x + 1];
            let br = frame[row2 + x + 2];

            if sobel_gradient_sq(tl, tc, tr, ml, mr, bl, bc, br) >= min_energy_sq {
                count += 1;
                if count >= min_y {
                    *is_active = true;
                    break;
                }
            }
        }
    }
}

fn classify_columns_across_frames(
    frames: &[FrameInfo<'_>],
    columns: &[ColumnSource],
    height: usize,
    min_mel: usize,
    min_y: usize,
    min_energy_sq: f64,
    output: &mut [bool],
) {
    if min_y == 0 {
        output.fill(true);
        return;
    }

    let start_y = min_mel.min(height - 2);

    for (x, is_active) in output.iter_mut().enumerate() {
        let c0 = columns[x];
        let c1 = columns[x + 1];
        let c2 = columns[x + 2];
        let f0 = &frames[c0.frame_index];
        let f1 = &frames[c1.frame_index];
        let f2 = &frames[c2.frame_index];
        let mut count = 0;

        for y in start_y..(height - 2) {
            let row00 = (y * f0.width) + c0.local_x;
            let row01 = (y * f1.width) + c1.local_x;
            let row02 = (y * f2.width) + c2.local_x;
            let row10 = row00 + f0.width;
            let row12 = row02 + f2.width;
            let row20 = row10 + f0.width;
            let row21 = row01 + (2 * f1.width);
            let row22 = row12 + f2.width;

            let tl = f0.data[row00];
            let tc = f1.data[row01];
            let tr = f2.data[row02];
            let ml = f0.data[row10];
            let mr = f2.data[row12];
            let bl = f0.data[row20];
            let bc = f1.data[row21];
            let br = f2.data[row22];

            if sobel_gradient_sq(tl, tc, tr, ml, mr, bl, bc, br) >= min_energy_sq {
                count += 1;
                if count >= min_y {
                    *is_active = true;
                    break;
                }
            }
        }
    }
}

#[inline]
fn sobel_gradient_sq(
    tl: f64,
    tc: f64,
    tr: f64,
    ml: f64,
    mr: f64,
    bl: f64,
    bc: f64,
    br: f64,
) -> f64 {
    let gradient_x = (tr + (2.0 * mr) + br) - (tl + (2.0 * ml) + bl);
    let gradient_y = (bl + (2.0 * bc) + br) - (tl + (2.0 * tc) + tr);
    (gradient_x * gradient_x) + (gradient_y * gradient_y)
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
        let min_x = 10;
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

            dbg!(file_path);
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

    #[ignore]
    #[test]
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
        eprintln!("test_vad_debug elapsed={elapsed}ms");
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

        let frames = to_array2(&dequantized_mel, n_mels);

        let edge_info = vad_boundaries(&[frames.clone()], &settings);

        let elapsed = start.elapsed().as_millis();
        eprintln!("test_vad_boundaries elapsed={elapsed}ms");
        let img = as_image(
            &[frames.clone()],
            &edge_info.non_intersected(),
            &edge_info.gradient_positions(),
        );

        img.save("./doc/vad.png").unwrap();
    }

    #[ignore]
    #[test]
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
        eprintln!("test_stage elapsed={elapsed}ms");
    }
}
