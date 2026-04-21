use mel_spec::quant::{load_tga_8bit, to_array2};
use mel_spec::vad::{vad_boundaries, DetectionSettings, VoiceActivityDetector};
use ndarray::{concatenate, s, Array, Array2, Axis};
use std::collections::HashSet;

struct LegacyVoiceActivityDetector {
    mel_buffer: Vec<Array2<f64>>,
    settings: DetectionSettings,
    idx: usize,
}

impl LegacyVoiceActivityDetector {
    fn new(settings: &DetectionSettings) -> Self {
        Self {
            mel_buffer: Vec::new(),
            settings: *settings,
            idx: 0,
        }
    }

    fn add(&mut self, frame: &Array2<f64>) -> Option<bool> {
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

        let window = &self.mel_buffer[self.idx - min_x..];
        let edge_info = legacy_vad_boundaries(window, &self.settings);
        let intersected = edge_info.intersected();
        if intersected.is_empty() {
            Some(false)
        } else {
            Some(intersected[0] == 0)
        }
    }
}

#[derive(Debug)]
struct LegacyEdgeInfo {
    non_intersected_columns: Vec<usize>,
    intersected_columns: Vec<usize>,
}

impl LegacyEdgeInfo {
    fn new(non_intersected_columns: Vec<usize>, intersected_columns: Vec<usize>) -> Self {
        Self {
            non_intersected_columns,
            intersected_columns,
        }
    }

    fn non_intersected(&self) -> Vec<usize> {
        self.non_intersected_columns.clone()
    }

    fn intersected(&self) -> Vec<usize> {
        self.intersected_columns.clone()
    }
}

fn legacy_vad_boundaries(frames: &[Array2<f64>], settings: &DetectionSettings) -> LegacyEdgeInfo {
    let array_views: Vec<_> = frames.iter().map(|a| a.view()).collect();
    let min_energy = settings.min_energy;
    let min_y = settings.min_y;
    let min_mel = settings.min_mel;

    let merged_frames = concatenate(Axis(1), &array_views).unwrap();
    let shape = merged_frames.raw_dim();
    let width = shape[1];
    let height = shape[0];

    let sobel_x =
        Array::from_shape_vec((3, 3), vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0])
            .unwrap();
    let sobel_y =
        Array::from_shape_vec((3, 3), vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0])
            .unwrap();

    let gradient_mag = Array::from_shape_fn((height - 2, width - 2), |(y, x)| {
        let view = merged_frames.slice(s![y..y + 3, x..x + 3]);
        let mut gradient_x = 0.0;
        let mut gradient_y = 0.0;
        for j in 0..3 {
            for i in 0..3 {
                gradient_x += view[[j, i]] * sobel_x[[j, i]];
                gradient_y += view[[j, i]] * sobel_y[[j, i]];
            }
        }
        (gradient_x * gradient_x + gradient_y * gradient_y).sqrt()
    });

    let mut raw_classification = Vec::with_capacity(width - 2);
    for x in 0..(width - 2) {
        let mut count = 0;
        for y in 0..(height - 2) {
            let grad = gradient_mag[(y, x)];
            if y >= min_mel && grad >= min_energy {
                count += 1;
            }
        }
        raw_classification.push(count >= min_y);
    }

    let smoothed_classification = legacy_smooth_mask(&raw_classification, 4);

    let mut intersected_columns = Vec::new();
    let mut non_intersected_columns = Vec::new();
    for (x, &active) in smoothed_classification.iter().enumerate() {
        if active {
            intersected_columns.push(x);
        } else {
            non_intersected_columns.push(x);
        }
    }

    LegacyEdgeInfo::new(non_intersected_columns, intersected_columns)
}

fn legacy_smooth_mask(mask: &[bool], window: usize) -> Vec<bool> {
    let n = mask.len();
    let mut smoothed = vec![false; n];
    for i in 0..n {
        let start = if i < window { 0 } else { i - window };
        let end = if i + window + 1 > n {
            n
        } else {
            i + window + 1
        };
        let count_true = mask[start..end].iter().filter(|&&val| val).count();
        if count_true * 2 >= (end - start) {
            smoothed[i] = true;
        }
    }
    smoothed
}

#[test]
fn vad_boundaries_matches_legacy_on_reference_fixtures() {
    let n_mels = 80;
    let cases = [
        (
            DetectionSettings {
                min_energy: 1.0,
                min_y: 3,
                min_x: 6,
                min_mel: 0,
            },
            "./testdata/quantized_mel_golden.tga",
        ),
        (
            DetectionSettings {
                min_energy: 1.0,
                min_y: 10,
                min_x: 10,
                min_mel: 0,
            },
            "./testdata/blank/frame_23760.tga",
        ),
        (
            DetectionSettings {
                min_energy: 1.0,
                min_y: 10,
                min_x: 10,
                min_mel: 0,
            },
            "./testdata/speech/frame_27125.tga",
        ),
        (
            DetectionSettings {
                min_energy: 1.0,
                min_y: 6,
                min_x: 1,
                min_mel: 0,
            },
            "./testdata/jfk_full_speech_chunk0_golden.tga",
        ),
    ];

    for (settings, path) in cases {
        let dequantized_mel = load_tga_8bit(path).unwrap();
        let frames = to_array2(&dequantized_mel, n_mels);

        let current = vad_boundaries(&[frames.clone()], &settings);
        let legacy = legacy_vad_boundaries(&[frames], &settings);

        assert_eq!(current.intersected(), legacy.intersected(), "{path}");
        assert_eq!(current.non_intersected(), legacy.non_intersected(), "{path}");
        assert_eq!(current.gradient_positions(), HashSet::new(), "{path}");
    }
}

#[test]
fn streaming_vad_matches_legacy_on_quantized_fixture() {
    let n_mels = 80;
    let settings = DetectionSettings {
        min_energy: 1.0,
        min_y: 3,
        min_x: 3,
        min_mel: 0,
    };

    let dequantized_mel = load_tga_8bit("./testdata/quantized_mel_golden.tga").unwrap();
    let frames = to_array2(&dequantized_mel, n_mels);
    let chunks: Vec<Array2<f64>> = frames
        .axis_chunks_iter(Axis(1), 1)
        .map(|chunk| chunk.to_owned())
        .collect();

    let mut current = VoiceActivityDetector::new(&settings);
    let mut legacy = LegacyVoiceActivityDetector::new(&settings);

    let current_outputs: Vec<Option<bool>> = chunks.iter().map(|chunk| current.add(chunk)).collect();
    let legacy_outputs: Vec<Option<bool>> = chunks.iter().map(|chunk| legacy.add(chunk)).collect();

    assert_eq!(current_outputs, legacy_outputs);
}
