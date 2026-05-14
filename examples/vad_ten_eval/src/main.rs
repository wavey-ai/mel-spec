use mel_spec::mel::{log_mel_spectrogram, mel, norm_mel};
use mel_spec::stft::Spectrogram;
use mel_spec::vad::{DetectionSettings, VadFrameTiming, VoiceActivityDetector};
use ndarray::Array1;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
struct Args {
    testset: PathBuf,
    fft_size: usize,
    hop_size: usize,
    n_mels: usize,
    min_energy: f64,
    min_y: usize,
    min_x: usize,
    min_mel: usize,
    time_mode: TimeMode,
    min_leading_active_columns: usize,
    min_active_columns: usize,
    min_confidence: f64,
    min_speech_ms: usize,
    merge_gap_ms: usize,
    print_segments: bool,
    max_files: Option<usize>,
}

#[derive(Debug, Copy, Clone)]
enum TimeMode {
    Start,
    Center,
    End,
}

#[derive(Debug, Copy, Clone)]
struct LabelSegment {
    start: f64,
    end: f64,
    speech: bool,
}

#[derive(Debug, Copy, Clone)]
struct TimedDecision {
    time_s: f64,
    speech: bool,
}

#[derive(Default, Debug, Copy, Clone)]
struct Metrics {
    tp: u64,
    fp: u64,
    tn: u64,
    fn_: u64,
}

impl Metrics {
    fn add(&mut self, predicted: bool, expected: bool) {
        match (predicted, expected) {
            (true, true) => self.tp += 1,
            (true, false) => self.fp += 1,
            (false, false) => self.tn += 1,
            (false, true) => self.fn_ += 1,
        }
    }

    fn merge(&mut self, other: Metrics) {
        self.tp += other.tp;
        self.fp += other.fp;
        self.tn += other.tn;
        self.fn_ += other.fn_;
    }

    fn total(&self) -> u64 {
        self.tp + self.fp + self.tn + self.fn_
    }

    fn precision(&self) -> f64 {
        ratio(self.tp, self.tp + self.fp)
    }

    fn recall(&self) -> f64 {
        ratio(self.tp, self.tp + self.fn_)
    }

    fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    fn accuracy(&self) -> f64 {
        ratio(self.tp + self.tn, self.total())
    }

    fn fpr(&self) -> f64 {
        ratio(self.fp, self.fp + self.tn)
    }

    fn fnr(&self) -> f64 {
        ratio(self.fn_, self.fn_ + self.tp)
    }

    fn speech_expected(&self) -> u64 {
        self.tp + self.fn_
    }

    fn speech_predicted(&self) -> u64 {
        self.tp + self.fp
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let settings = DetectionSettings {
        min_energy: args.min_energy,
        min_y: args.min_y,
        min_x: args.min_x,
        min_mel: args.min_mel,
    };

    let mut wavs = wav_files(&args.testset)?;
    if let Some(max_files) = args.max_files {
        wavs.truncate(max_files);
    }

    if wavs.is_empty() {
        return Err(format!("no .wav files found in {}", args.testset.display()).into());
    }

    println!("testset={}", args.testset.display());
    println!(
        "settings fft_size={} hop_size={} n_mels={} min_energy={} min_y={} min_x={} min_mel={} time_mode={:?}",
        args.fft_size,
        args.hop_size,
        args.n_mels,
        args.min_energy,
        args.min_y,
        args.min_x,
        args.min_mel,
        args.time_mode
    );
    println!(
        "confidence min_active_columns={} min_confidence={}",
        args.min_active_columns, args.min_confidence
    );
    println!(
        "boundary min_leading_active_columns={}",
        args.min_leading_active_columns
    );
    println!(
        "postprocess min_speech_ms={} merge_gap_ms={}",
        args.min_speech_ms, args.merge_gap_ms
    );
    println!();

    let mut total = Metrics::default();
    let mut file_rows = Vec::new();

    for wav_path in wavs {
        let label_path = wav_path.with_extension("scv");
        let labels = read_labels(&label_path)?;
        let (samples, sample_rate) = read_wav_mono_f32(&wav_path)?;
        let started = Instant::now();
        let (metrics, decisions) = evaluate_file(&samples, sample_rate, &labels, &args, &settings)?;
        let wall_s = started.elapsed().as_secs_f64();
        total.merge(metrics);

        let duration_s = samples.len() as f64 / sample_rate as f64;
        file_rows.push((wav_path.clone(), duration_s, wall_s, metrics));

        if args.print_segments {
            let predicted =
                decisions_to_segments(&decisions, args.hop_size as f64 / sample_rate as f64);
            println!(
                "segments {}",
                wav_path.file_name().unwrap().to_string_lossy()
            );
            for segment in predicted {
                println!("  {:.3},{:.3}", segment.start, segment.end);
            }
        }
    }

    println!("=== Aggregate ===");
    print_metrics("all", total);
    print_macro_metrics(&file_rows);
    print_speed_metrics(&file_rows);
    println!(
        "expected_speech_frames={} predicted_speech_frames={} total_frames={}",
        total.speech_expected(),
        total.speech_predicted(),
        total.total()
    );

    println!();
    println!("=== Per File ===");
    println!(
        "file,duration_s,wall_ms,rtf,rtfx,frames,precision,recall,f1,accuracy,fpr,fnr,tp,fp,tn,fn"
    );
    for (path, duration_s, wall_s, metrics) in file_rows {
        println!(
            "{},{:.3},{:.3},{:.6},{:.2},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{}",
            path.file_name().unwrap().to_string_lossy(),
            duration_s,
            wall_s * 1000.0,
            safe_ratio(wall_s, duration_s),
            safe_ratio(duration_s, wall_s),
            metrics.total(),
            metrics.precision(),
            metrics.recall(),
            metrics.f1(),
            metrics.accuracy(),
            metrics.fpr(),
            metrics.fnr(),
            metrics.tp,
            metrics.fp,
            metrics.tn,
            metrics.fn_
        );
    }

    Ok(())
}

fn evaluate_file(
    samples: &[f32],
    sample_rate: u32,
    labels: &[LabelSegment],
    args: &Args,
    settings: &DetectionSettings,
) -> Result<(Metrics, Vec<TimedDecision>), Box<dyn Error>> {
    let filters = mel(
        sample_rate as f64,
        args.fft_size,
        args.n_mels,
        None,
        None,
        false,
        true,
    );
    let stft_frames = Spectrogram::compute_all_cpu(samples, args.fft_size, args.hop_size);
    let timing = VadFrameTiming::new(args.fft_size, args.hop_size, sample_rate as f64);
    let mut vad = VoiceActivityDetector::new_with_timing(settings, timing);
    let mut decisions = Vec::new();

    for frame in stft_frames {
        let fft = Array1::from_vec(frame);
        let mel_frame = norm_mel(&log_mel_spectrogram(&fft, &filters));
        let Some(activity) = vad.add_activity(&mel_frame) else {
            continue;
        };

        let timestamps = activity
            .timestamps
            .expect("timestamped VAD should emit timestamps");
        let time_ms = match args.time_mode {
            TimeMode::Start => timestamps.start_ms,
            TimeMode::Center => timestamps.center_ms,
            TimeMode::End => timestamps.end_ms,
        };
        let time_s = time_ms as f64 / 1000.0;
        let speech = activity.active
            && activity.leading_active_columns >= args.min_leading_active_columns
            && activity.active_columns >= args.min_active_columns
            && activity.confidence >= args.min_confidence;
        decisions.push(TimedDecision { time_s, speech });
    }

    let frame_period_s = args.hop_size as f64 / sample_rate as f64;
    let decisions = postprocess_decisions(&decisions, frame_period_s, args);
    let mut metrics = Metrics::default();
    for decision in &decisions {
        let expected = label_at(labels, decision.time_s);
        metrics.add(decision.speech, expected);
    }

    Ok((metrics, decisions))
}

fn read_wav_mono_f32(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(format!("{} is not mono", path.display()).into());
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(format!("{} is not 16-bit PCM", path.display()).into());
    }

    let mut samples = Vec::with_capacity(reader.duration() as usize);
    for sample in reader.samples::<i16>() {
        samples.push(sample? as f32 / 32768.0);
    }

    Ok((samples, spec.sample_rate))
}

fn read_labels(path: &Path) -> Result<Vec<LabelSegment>, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let fields: Vec<&str> = content.trim().split(',').collect();
    if fields.len() < 4 || (fields.len() - 1) % 3 != 0 {
        return Err(format!("bad label file format: {}", path.display()).into());
    }

    let mut segments = Vec::new();
    for chunk in fields[1..].chunks_exact(3) {
        segments.push(LabelSegment {
            start: chunk[0].parse()?,
            end: chunk[1].parse()?,
            speech: chunk[2].parse::<u8>()? == 1,
        });
    }

    Ok(segments)
}

fn wav_files(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut wavs = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "wav") {
            wavs.push(path);
        }
    }
    wavs.sort();
    Ok(wavs)
}

fn label_at(labels: &[LabelSegment], time_s: f64) -> bool {
    labels
        .iter()
        .find(|segment| time_s >= segment.start && time_s < segment.end)
        .is_some_and(|segment| segment.speech)
}

fn decisions_to_segments(decisions: &[TimedDecision], half_width_s: f64) -> Vec<LabelSegment> {
    let mut out = Vec::new();
    let mut open_start = None;

    for decision in decisions {
        match (decision.speech, open_start) {
            (true, None) => open_start = Some((decision.time_s - half_width_s / 2.0).max(0.0)),
            (false, Some(start)) => {
                out.push(LabelSegment {
                    start,
                    end: (decision.time_s - half_width_s / 2.0).max(start),
                    speech: true,
                });
                open_start = None;
            }
            _ => {}
        }
    }

    if let (Some(start), Some(last)) = (open_start, decisions.last()) {
        out.push(LabelSegment {
            start,
            end: last.time_s + half_width_s / 2.0,
            speech: true,
        });
    }

    out
}

fn postprocess_decisions(
    decisions: &[TimedDecision],
    frame_period_s: f64,
    args: &Args,
) -> Vec<TimedDecision> {
    if decisions.is_empty() || (args.min_speech_ms == 0 && args.merge_gap_ms == 0) {
        return decisions.to_vec();
    }

    let mut segments = decisions_to_segments(decisions, frame_period_s);
    if args.merge_gap_ms > 0 {
        let merge_gap_s = args.merge_gap_ms as f64 / 1000.0;
        segments = merge_close_segments(&segments, merge_gap_s);
    }
    if args.min_speech_ms > 0 {
        let min_speech_s = args.min_speech_ms as f64 / 1000.0;
        segments.retain(|segment| segment.end - segment.start >= min_speech_s);
    }

    decisions
        .iter()
        .map(|decision| TimedDecision {
            time_s: decision.time_s,
            speech: segments
                .iter()
                .any(|segment| decision.time_s >= segment.start && decision.time_s < segment.end),
        })
        .collect()
}

fn merge_close_segments(segments: &[LabelSegment], merge_gap_s: f64) -> Vec<LabelSegment> {
    let mut merged: Vec<LabelSegment> = Vec::new();
    for segment in segments {
        let Some(last) = merged.last_mut() else {
            merged.push(*segment);
            continue;
        };

        if segment.start - last.end <= merge_gap_s {
            last.end = last.end.max(segment.end);
        } else {
            merged.push(*segment);
        }
    }
    merged
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args {
        testset: PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/ten-vad"
        )),
        fft_size: 400,
        hop_size: 160,
        n_mels: 80,
        min_energy: 1.0,
        min_y: 10,
        min_x: 5,
        min_mel: 0,
        time_mode: TimeMode::Center,
        min_leading_active_columns: 1,
        min_active_columns: 1,
        min_confidence: 0.0,
        min_speech_ms: 100,
        merge_gap_ms: 100,
        print_segments: false,
        max_files: None,
    };

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--testset" => args.testset = PathBuf::from(next_value(&mut iter, &arg)?),
            "--fft-size" => args.fft_size = next_value(&mut iter, &arg)?.parse()?,
            "--hop-size" => args.hop_size = next_value(&mut iter, &arg)?.parse()?,
            "--n-mels" => args.n_mels = next_value(&mut iter, &arg)?.parse()?,
            "--min-energy" => args.min_energy = next_value(&mut iter, &arg)?.parse()?,
            "--min-y" => args.min_y = next_value(&mut iter, &arg)?.parse()?,
            "--min-x" => args.min_x = next_value(&mut iter, &arg)?.parse()?,
            "--min-mel" => args.min_mel = next_value(&mut iter, &arg)?.parse()?,
            "--min-leading-active-columns" => {
                args.min_leading_active_columns = next_value(&mut iter, &arg)?.parse()?
            }
            "--min-active-columns" => {
                args.min_active_columns = next_value(&mut iter, &arg)?.parse()?
            }
            "--min-confidence" => args.min_confidence = next_value(&mut iter, &arg)?.parse()?,
            "--max-files" => args.max_files = Some(next_value(&mut iter, &arg)?.parse()?),
            "--min-speech-ms" => args.min_speech_ms = next_value(&mut iter, &arg)?.parse()?,
            "--merge-gap-ms" => args.merge_gap_ms = next_value(&mut iter, &arg)?.parse()?,
            "--time-mode" => {
                args.time_mode = match next_value(&mut iter, &arg)?.as_str() {
                    "start" => TimeMode::Start,
                    "center" => TimeMode::Center,
                    "end" => TimeMode::End,
                    other => return Err(format!("unknown --time-mode {other}").into()),
                };
            }
            "--print-segments" => args.print_segments = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(args)
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn print_help() {
    println!(
        "vad_ten_eval [options]\n\
         \n\
         Options:\n\
           --testset PATH       TEN-VAD testset directory, default ../../testdata/ten-vad\n\
           --fft-size N         STFT size, default 400\n\
           --hop-size N         STFT hop size, default 160\n\
           --n-mels N           Mel bins for VAD, default 80\n\
           --min-energy F       mel-spec VAD min_energy, default 1.0\n\
           --min-y N            mel-spec VAD min_y, default 10\n\
           --min-x N            mel-spec VAD min_x, default 5\n\
           --min-mel N          mel-spec VAD min_mel, default 0\n\
           --time-mode MODE     start, center, or end, default center\n\
           --min-leading-active-columns N  Require N contiguous active columns from the boundary\n\
           --min-active-columns N  Require at least N active columns in the VAD window\n\
           --min-confidence F   Require active/window column ratio >= F\n\
           --min-speech-ms N    Drop predicted speech segments shorter than N ms, default 100\n\
           --merge-gap-ms N     Merge predicted speech segments separated by <= N ms, default 100\n\
           --max-files N        Evaluate only the first N wavs\n\
           --print-segments     Print predicted speech timestamp segments"
    );
}

fn print_metrics(label: &str, metrics: Metrics) {
    println!(
        "{label}: frames={} precision={:.4} recall={:.4} f1={:.4} accuracy={:.4} fpr={:.4} fnr={:.4} tp={} fp={} tn={} fn={}",
        metrics.total(),
        metrics.precision(),
        metrics.recall(),
        metrics.f1(),
        metrics.accuracy(),
        metrics.fpr(),
        metrics.fnr(),
        metrics.tp,
        metrics.fp,
        metrics.tn,
        metrics.fn_
    );
}

fn print_macro_metrics(file_rows: &[(PathBuf, f64, f64, Metrics)]) {
    let count = file_rows.len() as f64;
    let (precision, recall, f1, accuracy, fpr, fnr) = file_rows.iter().fold(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        |(precision, recall, f1, accuracy, fpr, fnr), (_, _, _, metrics)| {
            (
                precision + metrics.precision(),
                recall + metrics.recall(),
                f1 + metrics.f1(),
                accuracy + metrics.accuracy(),
                fpr + metrics.fpr(),
                fnr + metrics.fnr(),
            )
        },
    );

    println!(
        "macro: files={} precision={:.4} recall={:.4} f1={:.4} accuracy={:.4} fpr={:.4} fnr={:.4}",
        file_rows.len(),
        precision / count,
        recall / count,
        f1 / count,
        accuracy / count,
        fpr / count,
        fnr / count
    );
}

fn print_speed_metrics(file_rows: &[(PathBuf, f64, f64, Metrics)]) {
    let audio_s: f64 = file_rows
        .iter()
        .map(|(_, duration_s, _, _)| duration_s)
        .sum();
    let wall_s: f64 = file_rows.iter().map(|(_, _, wall_s, _)| wall_s).sum();

    println!(
        "speed: audio_s={:.3} wall_s={:.3} rtf={:.6} rtfx={:.2}",
        audio_s,
        wall_s,
        safe_ratio(wall_s, audio_s),
        safe_ratio(audio_s, wall_s)
    );
}

fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn safe_ratio(num: f64, den: f64) -> f64 {
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}
