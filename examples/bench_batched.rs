//! Benchmark comparing batched CPU vs batched GPU mel spectrogram
//!
//! Run with: cargo run --release --features cuda --example bench_batched

use std::time::Instant;
use mel_spec::stft::Spectrogram;

fn main() {
    let sample_rate = 16000usize;
    let fft_size = 512;
    let hop_size = 160;
    let n_mels = 80;
    let iterations = 100;

    println!("Benchmarking full mel spectrogram pipeline (STFT + mel filterbank + log)");
    println!("FFT size: {}, Hop size: {}, Mel bins: {}\n", fft_size, hop_size, n_mels);

    // Test different audio lengths
    for duration_secs in [10, 30, 60, 120, 300] {
        let num_samples = duration_secs * sample_rate;

        // Generate test audio (sine wave)
        let audio: Vec<f32> = (0..num_samples)
            .map(|i| ((i as f32 * 0.01).sin() * 0.1))
            .collect();

        let expected_frames = (num_samples - fft_size) / hop_size + 1;

        // Warmup CPU
        for _ in 0..3 {
            let _ = Spectrogram::compute_mel_spectrogram_cpu(
                &audio, fft_size, hop_size, n_mels, sample_rate as f64
            );
        }

        // Benchmark CPU
        let start = Instant::now();
        for _ in 0..iterations {
            let frames = Spectrogram::compute_mel_spectrogram_cpu(
                &audio, fft_size, hop_size, n_mels, sample_rate as f64
            );
            std::hint::black_box(frames);
        }
        let cpu_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

        #[cfg(feature = "cuda")]
        {
            // Warmup GPU
            for _ in 0..3 {
                let _ = Spectrogram::compute_mel_spectrogram_cuda(
                    &audio, fft_size, hop_size, n_mels, sample_rate as f64
                );
            }

            // Benchmark GPU
            let start = Instant::now();
            for _ in 0..iterations {
                match Spectrogram::compute_mel_spectrogram_cuda(
                    &audio, fft_size, hop_size, n_mels, sample_rate as f64
                ) {
                    Ok(frames) => { std::hint::black_box(frames); }
                    Err(e) => {
                        println!("GPU error: {:?}", e);
                        return;
                    }
                }
            }
            let gpu_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            let speedup = cpu_time / gpu_time;
            println!(
                "{}s audio ({} frames): CPU {:.2}ms, GPU {:.2}ms, speedup {:.2}x {}",
                duration_secs,
                expected_frames,
                cpu_time,
                gpu_time,
                speedup,
                if speedup > 1.0 { "(GPU faster)" } else { "(CPU faster)" }
            );

            // Verify correctness for first test
            if duration_secs == 10 {
                let cpu_frames = Spectrogram::compute_mel_spectrogram_cpu(
                    &audio, fft_size, hop_size, n_mels, sample_rate as f64
                );
                let gpu_frames = Spectrogram::compute_mel_spectrogram_cuda(
                    &audio, fft_size, hop_size, n_mels, sample_rate as f64
                ).unwrap();

                assert_eq!(cpu_frames.len(), gpu_frames.len(), "Frame count mismatch");

                let mut max_diff = 0.0f64;
                for (cpu_frame, gpu_frame) in cpu_frames.iter().zip(gpu_frames.iter()) {
                    for (c, g) in cpu_frame.iter().zip(gpu_frame.iter()) {
                        let diff = (c - g).abs();
                        max_diff = max_diff.max(diff);
                    }
                }
                println!("  Max CPU/GPU difference: {:.2e}", max_diff);
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            println!(
                "{}s audio ({} frames): CPU {:.2}ms (GPU benchmark skipped - cuda feature not enabled)",
                duration_secs,
                expected_frames,
                cpu_time
            );
        }
    }

    // Data transfer comparison
    println!("\n--- Data transfer comparison ---");
    let duration_secs = 300;
    let num_samples = duration_secs * sample_rate;
    let num_frames = (num_samples - fft_size) / hop_size + 1;

    let fft_output_bytes = num_frames * fft_size * 2 * 8; // complex f64
    let mel_output_bytes = num_frames * n_mels * 8; // f64

    println!("For {}s audio ({} frames):", duration_secs, num_frames);
    println!("  FFT-only output:    {:.1} MB (what old benchmark transferred)",
             fft_output_bytes as f64 / 1_000_000.0);
    println!("  Mel spec output:    {:.1} MB (what new benchmark transfers)",
             mel_output_bytes as f64 / 1_000_000.0);
    println!("  Reduction:          {:.1}x less data",
             fft_output_bytes as f64 / mel_output_bytes as f64);
}
