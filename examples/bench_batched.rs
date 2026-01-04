//! Benchmark comparing batched CPU vs batched GPU FFT
//!
//! Run with: cargo run --release --features cuda --example bench_batched

use std::time::Instant;
use mel_spec::stft::Spectrogram;

fn main() {
    let duration_secs = 10;
    let sample_rate = 16000usize;
    let num_samples = duration_secs * sample_rate;

    // Generate test audio (sine wave)
    let audio: Vec<f32> = (0..num_samples)
        .map(|i| ((i as f32 * 0.01).sin() * 0.1))
        .collect();

    let fft_size = 512;
    let hop_size = 160;
    let iterations = 100;

    println!("Benchmarking STFT on {}s audio ({} samples)", duration_secs, num_samples);
    println!("FFT size: {}, Hop size: {}", fft_size, hop_size);

    // Expected frames
    let expected_frames = (num_samples - fft_size) / hop_size + 1;
    println!("Expected frames: {}\n", expected_frames);

    // Warmup CPU
    for _ in 0..5 {
        let _ = Spectrogram::compute_all_cpu(&audio, fft_size, hop_size);
    }

    // Benchmark CPU (batched)
    let start = Instant::now();
    for _ in 0..iterations {
        let frames = Spectrogram::compute_all_cpu(&audio, fft_size, hop_size);
        std::hint::black_box(frames);
    }
    let cpu_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;
    println!("CPU (batched): {:.3} ms per {}s audio", cpu_time, duration_secs);

    // Benchmark GPU (batched)
    #[cfg(feature = "cuda")]
    {
        // Warmup GPU
        for _ in 0..5 {
            let _ = Spectrogram::compute_all_cuda(&audio, fft_size, hop_size);
        }

        let start = Instant::now();
        for _ in 0..iterations {
            match Spectrogram::compute_all_cuda(&audio, fft_size, hop_size) {
                Ok(frames) => { std::hint::black_box(frames); }
                Err(e) => {
                    println!("GPU error: {:?}", e);
                    return;
                }
            }
        }
        let gpu_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;
        println!("GPU (batched): {:.3} ms per {}s audio", gpu_time, duration_secs);

        if gpu_time > 0.0 {
            let speedup = cpu_time / gpu_time;
            println!("\nSpeedup: {:.2}x", speedup);
            if speedup > 1.0 {
                println!("GPU is faster!");
            } else {
                println!("CPU is faster (GPU overhead not amortized)");
            }
        }

        // Verify correctness
        let cpu_frames = Spectrogram::compute_all_cpu(&audio, fft_size, hop_size);
        let gpu_frames = Spectrogram::compute_all_cuda(&audio, fft_size, hop_size).unwrap();

        assert_eq!(cpu_frames.len(), gpu_frames.len(), "Frame count mismatch");

        let mut max_diff = 0.0f64;
        for (cpu_frame, gpu_frame) in cpu_frames.iter().zip(gpu_frames.iter()) {
            for (c, g) in cpu_frame.iter().zip(gpu_frame.iter()) {
                let diff = ((c.re - g.re).abs()).max((c.im - g.im).abs());
                max_diff = max_diff.max(diff);
            }
        }
        println!("\nMax CPU/GPU difference: {:.2e}", max_diff);
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("GPU benchmark skipped (cuda feature not enabled)");
    }
}
