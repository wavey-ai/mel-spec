use crate::stft::{frame_windows, hann_window};
use std::ffi::c_void;
use std::fmt;
use std::mem::size_of;
use std::ptr;
use std::slice;

const TARGET_BATCH_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug)]
pub enum CudaError {
    Runtime(String),
    Unavailable(&'static str),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(msg) => write!(f, "CUDA error: {msg}"),
            Self::Unavailable(msg) => write!(f, "CUDA unavailable: {msg}"),
        }
    }
}

impl std::error::Error for CudaError {}

pub struct CudaMelSpectrogram {
    d_filters: *mut c_void,
    d_mel_out: *mut c_void,
    fft_size: usize,
    h_mel_out: *mut f64,
    hop_size: usize,
    max_frames_per_batch: usize,
    n_mels: usize,
    plan: ffi::CudaPlan,
}

impl CudaMelSpectrogram {
    pub fn new(
        fft_size: usize,
        hop_size: usize,
        sampling_rate: f64,
        n_mels: usize,
    ) -> Result<Self, CudaError> {
        if fft_size == 0 || hop_size == 0 || n_mels == 0 {
            return Err(CudaError::Unavailable(
                "fft_size, hop_size, and n_mels must be non-zero",
            ));
        }

        let max_frames_per_batch = max_frames_per_batch(fft_size, n_mels);
        let plan = ffi::CudaPlan::new_batch(fft_size, max_frames_per_batch)?;

        let filters = crate::mel::mel(sampling_rate, fft_size, n_mels, None, None, false, true)
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let filter_bytes = size_bytes::<f64>(filters.len());
        let d_filters = ffi::alloc_device(filter_bytes)?;
        ffi::memcpy_h2d_async(
            d_filters,
            filters.as_ptr() as *const c_void,
            filter_bytes,
            plan.stream(),
        )?;
        ffi::stream_sync(plan.stream())?;

        let mel_output_bytes = size_bytes::<f64>(max_frames_per_batch * n_mels);
        let d_mel_out = ffi::alloc_device(mel_output_bytes)?;
        let h_mel_out = ffi::alloc_host(mel_output_bytes)? as *mut f64;

        Ok(Self {
            d_filters,
            d_mel_out,
            fft_size,
            h_mel_out,
            hop_size,
            max_frames_per_batch,
            n_mels,
            plan,
        })
    }

    pub fn max_frames_per_batch(&self) -> usize {
        self.max_frames_per_batch
    }

    pub fn compute_mel_spectrogram(&mut self, samples: &[f32]) -> Result<Vec<Vec<f32>>, CudaError> {
        let window = hann_window(self.fft_size);
        let frames = frame_windows(samples, self.fft_size, self.hop_size, &window);
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let mut output = Vec::with_capacity(frames.len());
        for batch in frames.chunks(self.max_frames_per_batch) {
            output.extend(self.compute_batch(batch)?);
        }

        Ok(output)
    }

    fn compute_batch(&mut self, batch: &[Vec<f64>]) -> Result<Vec<Vec<f32>>, CudaError> {
        let frames = batch.len();
        let windowed_samples = batch
            .iter()
            .flat_map(|frame| frame.iter().copied())
            .collect::<Vec<_>>();

        self.plan.fft_device(&windowed_samples, frames)?;

        ffi::launch_mel(
            self.plan.device_ptr(),
            self.d_mel_out as *mut f64,
            self.d_filters as *const f64,
            frames,
            self.fft_size,
            self.n_mels,
            self.plan.stream(),
        )?;

        let mel_output_bytes = size_bytes::<f64>(frames * self.n_mels);
        ffi::memcpy_d2h_async(
            self.h_mel_out as *mut c_void,
            self.d_mel_out as *const c_void,
            mel_output_bytes,
            self.plan.stream(),
        )?;
        ffi::stream_sync(self.plan.stream())?;

        let host_mels = unsafe { slice::from_raw_parts(self.h_mel_out, frames * self.n_mels) };
        let mut output = Vec::with_capacity(frames);
        for frame in host_mels.chunks(self.n_mels) {
            let row = frame.iter().map(|value| *value as f32).collect::<Vec<_>>();
            output.push(crate::mel::norm_mel_vec(&row));
        }

        Ok(output)
    }
}

impl Drop for CudaMelSpectrogram {
    fn drop(&mut self) {
        ffi::free_device(self.d_filters);
        ffi::free_device(self.d_mel_out);
        ffi::free_host(self.h_mel_out as *mut c_void);
    }
}

fn max_frames_per_batch(fft_size: usize, n_mels: usize) -> usize {
    let fft_frame_bytes = size_bytes::<ffi::CufftDoubleComplex>(fft_size);
    let mel_frame_bytes = size_bytes::<f64>(n_mels);
    let bytes_per_frame = fft_frame_bytes + mel_frame_bytes;
    ((TARGET_BATCH_BYTES / bytes_per_frame).max(1) as usize).min(8192)
}

fn size_bytes<T>(len: usize) -> u64 {
    (size_of::<T>() * len) as u64
}

mod ffi {
    use super::{c_void, ptr, CudaError};

    const CUDA_SUCCESS: i32 = 0;
    const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
    const CUDA_HOST_ALLOC_DEFAULT: u32 = 0;

    pub const CUFFT_FORWARD: i32 = -1;
    const CUFFT_SUCCESS: i32 = 0;
    const CUFFT_Z2Z: i32 = 0x69;

    type CudaErrorCode = i32;
    type CufftResult = i32;
    type CufftHandle = i32;
    pub type CudaStream = *mut c_void;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CufftDoubleComplex {
        pub x: f64,
        pub y: f64,
    }

    unsafe extern "C" {
        fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> CudaErrorCode;
        fn cudaFree(dev_ptr: *mut c_void) -> CudaErrorCode;
        fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> CudaErrorCode;
        fn cudaFreeHost(ptr: *mut c_void) -> CudaErrorCode;
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: i32,
            stream: CudaStream,
        ) -> CudaErrorCode;
        fn cudaStreamCreate(stream: *mut CudaStream) -> CudaErrorCode;
        fn cudaStreamDestroy(stream: CudaStream) -> CudaErrorCode;
        fn cudaStreamSynchronize(stream: CudaStream) -> CudaErrorCode;

        fn cufftPlan1d(plan: *mut CufftHandle, nx: i32, fft_type: i32, batch: i32) -> CufftResult;
        fn cufftDestroy(plan: CufftHandle) -> CufftResult;
        fn cufftSetStream(plan: CufftHandle, stream: CudaStream) -> CufftResult;
        fn cufftExecZ2Z(
            plan: CufftHandle,
            input: *mut CufftDoubleComplex,
            output: *mut CufftDoubleComplex,
            direction: i32,
        ) -> CufftResult;

        fn launch_mel_kernel(
            fft: *const CufftDoubleComplex,
            out: *mut f64,
            filters: *const f64,
            frames: i32,
            fft_size: i32,
            n_mels: i32,
            stream: CudaStream,
        ) -> CudaErrorCode;
    }

    pub struct CudaPlan {
        batch: usize,
        d_buf: *mut CufftDoubleComplex,
        fft_size: usize,
        h_buf: *mut CufftDoubleComplex,
        plan: CufftHandle,
        stream: CudaStream,
    }

    impl CudaPlan {
        pub fn new_batch(fft_size: usize, batch: usize) -> Result<Self, CudaError> {
            let mut plan: CufftHandle = 0;
            let plan_result = unsafe {
                cufftPlan1d(
                    &mut plan as *mut CufftHandle,
                    fft_size as i32,
                    CUFFT_Z2Z,
                    batch as i32,
                )
            };
            if plan_result != CUFFT_SUCCESS {
                return Err(CudaError::Unavailable("cufftPlan1d failed"));
            }

            let mut stream: CudaStream = ptr::null_mut();
            let stream_result = unsafe { cudaStreamCreate(&mut stream as *mut CudaStream) };
            if stream_result != CUDA_SUCCESS {
                unsafe {
                    let _ = cufftDestroy(plan);
                }
                return Err(CudaError::Unavailable("cudaStreamCreate failed"));
            }

            let set_stream_result = unsafe { cufftSetStream(plan, stream) };
            if set_stream_result != CUFFT_SUCCESS {
                unsafe {
                    let _ = cudaStreamDestroy(stream);
                    let _ = cufftDestroy(plan);
                }
                return Err(CudaError::Unavailable("cufftSetStream failed"));
            }

            let mut d_buf: *mut CufftDoubleComplex = ptr::null_mut();
            let d_buf_result = unsafe {
                cudaMalloc(
                    &mut d_buf as *mut *mut CufftDoubleComplex as *mut *mut c_void,
                    fft_size * batch * std::mem::size_of::<CufftDoubleComplex>(),
                )
            };
            if d_buf_result != CUDA_SUCCESS {
                unsafe {
                    let _ = cudaStreamDestroy(stream);
                    let _ = cufftDestroy(plan);
                }
                return Err(CudaError::Unavailable("cudaMalloc failed"));
            }

            let mut h_buf: *mut CufftDoubleComplex = ptr::null_mut();
            let h_buf_result = unsafe {
                cudaHostAlloc(
                    &mut h_buf as *mut *mut CufftDoubleComplex as *mut *mut c_void,
                    fft_size * batch * std::mem::size_of::<CufftDoubleComplex>(),
                    CUDA_HOST_ALLOC_DEFAULT,
                )
            };
            if h_buf_result != CUDA_SUCCESS {
                unsafe {
                    let _ = cudaFree(d_buf as *mut c_void);
                    let _ = cudaStreamDestroy(stream);
                    let _ = cufftDestroy(plan);
                }
                return Err(CudaError::Unavailable("cudaHostAlloc failed"));
            }

            Ok(Self {
                batch,
                d_buf,
                fft_size,
                h_buf,
                plan,
                stream,
            })
        }

        pub fn stream(&self) -> CudaStream {
            self.stream
        }

        pub fn device_ptr(&self) -> *const c_void {
            self.d_buf as *const c_void
        }

        pub fn fft_device(
            &mut self,
            windowed_samples: &[f64],
            frames: usize,
        ) -> Result<(), CudaError> {
            if frames == 0 || frames > self.batch {
                return Err(CudaError::Runtime("invalid batch size".into()));
            }
            if windowed_samples.len() != frames * self.fft_size {
                return Err(CudaError::Runtime("input length mismatch".into()));
            }

            let host_slice =
                unsafe { std::slice::from_raw_parts_mut(self.h_buf, self.fft_size * self.batch) };
            for (i, chunk) in windowed_samples.chunks(self.fft_size).enumerate() {
                let offset = i * self.fft_size;
                for (j, &value) in chunk.iter().enumerate() {
                    host_slice[offset + j].x = value;
                    host_slice[offset + j].y = 0.0;
                }
            }
            if frames < self.batch {
                for idx in (frames * self.fft_size)..(self.batch * self.fft_size) {
                    host_slice[idx].x = 0.0;
                    host_slice[idx].y = 0.0;
                }
            }

            let byte_len = self.fft_size * self.batch * std::mem::size_of::<CufftDoubleComplex>();
            let copy_result = unsafe {
                cudaMemcpyAsync(
                    self.d_buf as *mut c_void,
                    host_slice.as_ptr() as *const c_void,
                    byte_len,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    self.stream,
                )
            };
            if copy_result != CUDA_SUCCESS {
                return Err(CudaError::Runtime("cudaMemcpy H2D failed".into()));
            }

            let exec_result =
                unsafe { cufftExecZ2Z(self.plan, self.d_buf, self.d_buf, CUFFT_FORWARD) };
            if exec_result != CUFFT_SUCCESS {
                return Err(CudaError::Runtime("cufftExecZ2Z failed".into()));
            }

            Ok(())
        }
    }

    impl Drop for CudaPlan {
        fn drop(&mut self) {
            unsafe {
                let _ = cufftDestroy(self.plan);
                let _ = cudaFree(self.d_buf as *mut c_void);
                let _ = cudaFreeHost(self.h_buf as *mut c_void);
                let _ = cudaStreamDestroy(self.stream);
            }
        }
    }

    pub fn alloc_device(size: u64) -> Result<*mut c_void, CudaError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let result = unsafe { cudaMalloc(&mut ptr as *mut *mut c_void, size as usize) };
        if result != CUDA_SUCCESS {
            return Err(CudaError::Runtime("cudaMalloc failed".into()));
        }
        Ok(ptr)
    }

    pub fn alloc_host(size: u64) -> Result<*mut c_void, CudaError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let result = unsafe {
            cudaHostAlloc(
                &mut ptr as *mut *mut c_void,
                size as usize,
                CUDA_HOST_ALLOC_DEFAULT,
            )
        };
        if result != CUDA_SUCCESS {
            return Err(CudaError::Runtime("cudaHostAlloc failed".into()));
        }
        Ok(ptr)
    }

    pub fn free_device(ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        unsafe {
            let _ = cudaFree(ptr);
        }
    }

    pub fn free_host(ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        unsafe {
            let _ = cudaFreeHost(ptr);
        }
    }

    pub fn memcpy_h2d_async(
        dst: *mut c_void,
        src: *const c_void,
        size: u64,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let result =
            unsafe { cudaMemcpyAsync(dst, src, size as usize, CUDA_MEMCPY_HOST_TO_DEVICE, stream) };
        if result != CUDA_SUCCESS {
            return Err(CudaError::Runtime("cudaMemcpyAsync H2D failed".into()));
        }
        Ok(())
    }

    pub fn memcpy_d2h_async(
        dst: *mut c_void,
        src: *const c_void,
        size: u64,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let result =
            unsafe { cudaMemcpyAsync(dst, src, size as usize, CUDA_MEMCPY_DEVICE_TO_HOST, stream) };
        if result != CUDA_SUCCESS {
            return Err(CudaError::Runtime("cudaMemcpyAsync D2H failed".into()));
        }
        Ok(())
    }

    pub fn stream_sync(stream: CudaStream) -> Result<(), CudaError> {
        let result = unsafe { cudaStreamSynchronize(stream) };
        if result != CUDA_SUCCESS {
            return Err(CudaError::Runtime("cudaStreamSynchronize failed".into()));
        }
        Ok(())
    }

    pub fn launch_mel(
        fft: *const c_void,
        out: *mut f64,
        filters: *const f64,
        frames: usize,
        fft_size: usize,
        n_mels: usize,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let result = unsafe {
            launch_mel_kernel(
                fft as *const CufftDoubleComplex,
                out,
                filters,
                frames as i32,
                fft_size as i32,
                n_mels as i32,
                stream,
            )
        };
        if result != CUDA_SUCCESS {
            return Err(CudaError::Runtime("launch_mel_kernel failed".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::Spectrogram;
    use std::time::Instant;

    #[test]
    fn cuda_matches_cpu_for_whisper_fft_400() {
        let fft_size = 400;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16_000.0;
        let samples = (0..16_000)
            .map(|i| {
                let t = i as f32 / sampling_rate as f32;
                0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.25 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.10 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                    + 0.05 * (2.0 * std::f32::consts::PI * 1760.0 * t).sin()
            })
            .collect::<Vec<_>>();

        let cpu = Spectrogram::compute_mel_spectrogram_cpu(
            &samples,
            fft_size,
            hop_size,
            n_mels,
            sampling_rate,
        );

        let mut cuda = match CudaMelSpectrogram::new(fft_size, hop_size, sampling_rate, n_mels) {
            Ok(cuda) => cuda,
            Err(err) => {
                eprintln!("Skipping cuda test: {err}");
                return;
            }
        };

        let gpu = cuda
            .compute_mel_spectrogram(&samples)
            .expect("cuda mel spectrogram");

        assert_eq!(cpu.len(), gpu.len(), "frame count mismatch");

        let mut max_delta = 0.0f32;
        let mut sum_delta = 0.0f32;
        let mut count = 0usize;
        for (cpu_frame, gpu_frame) in cpu.iter().zip(gpu.iter()) {
            assert_eq!(cpu_frame.len(), gpu_frame.len(), "mel width mismatch");
            for (&cpu_value, &gpu_value) in cpu_frame.iter().zip(gpu_frame.iter()) {
                let delta = (cpu_value - gpu_value).abs();
                max_delta = max_delta.max(delta);
                sum_delta += delta;
                count += 1;
            }
        }

        let mean_delta = sum_delta / count as f32;
        assert!(
            max_delta < 0.08,
            "max delta too large: {max_delta}, mean delta: {mean_delta}"
        );
        assert!(mean_delta < 0.01, "mean delta too large: {mean_delta}");
    }

    #[test]
    #[ignore]
    fn benchmark_cuda_vs_cpu_whisper_fft_400() {
        let fft_size = 400;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16_000.0;

        let startup_begin = Instant::now();
        let mut cuda = match CudaMelSpectrogram::new(fft_size, hop_size, sampling_rate, n_mels) {
            Ok(cuda) => cuda,
            Err(err) => {
                eprintln!("Skipping cuda benchmark: {err}");
                return;
            }
        };
        let startup_elapsed = startup_begin.elapsed();
        println!(
            "CUDA startup: {:.2} ms",
            startup_elapsed.as_secs_f64() * 1000.0
        );
        println!("CUDA batch size: {}", cuda.max_frames_per_batch());

        for seconds in [10usize, 60usize, 300usize] {
            let sample_count = seconds * sampling_rate as usize;
            let samples = (0..sample_count)
                .map(|i| {
                    let t = i as f32 / sampling_rate as f32;
                    0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                        + 0.25 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                        + 0.10 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                        + 0.05 * (2.0 * std::f32::consts::PI * 1760.0 * t).sin()
                })
                .collect::<Vec<_>>();

            let warmup = cuda
                .compute_mel_spectrogram(&samples)
                .expect("cuda warmup mel spectrogram");
            assert!(!warmup.is_empty(), "expected non-empty warmup output");

            let cpu_begin = Instant::now();
            let cpu = Spectrogram::compute_mel_spectrogram_cpu(
                &samples,
                fft_size,
                hop_size,
                n_mels,
                sampling_rate,
            );
            let cpu_elapsed = cpu_begin.elapsed();

            let cuda_begin = Instant::now();
            let cuda_out = cuda
                .compute_mel_spectrogram(&samples)
                .expect("cuda mel spectrogram");
            let cuda_elapsed = cuda_begin.elapsed();

            assert_eq!(cpu.len(), cuda_out.len(), "frame count mismatch");

            let speedup = cpu_elapsed.as_secs_f64() / cuda_elapsed.as_secs_f64().max(f64::EPSILON);
            println!(
                "{}s audio: CPU {:.2} ms, CUDA {:.2} ms, speedup x{:.2}",
                seconds,
                cpu_elapsed.as_secs_f64() * 1000.0,
                cuda_elapsed.as_secs_f64() * 1000.0,
                speedup
            );
        }
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    #[test]
    #[ignore]
    fn benchmark_cuda_vs_wgpu_whisper_fft_400() {
        use crate::wgpu::WgpuMelSpectrogram;

        let fft_size = 400;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16_000.0;

        let mut cuda = match CudaMelSpectrogram::new(fft_size, hop_size, sampling_rate, n_mels) {
            Ok(cuda) => cuda,
            Err(err) => {
                eprintln!("Skipping cuda benchmark: {err}");
                return;
            }
        };
        let wgpu = match WgpuMelSpectrogram::new(fft_size, hop_size, sampling_rate, n_mels) {
            Ok(wgpu) => Some(wgpu),
            Err(err) => {
                eprintln!("Skipping wgpu portion of mixed benchmark: {err}");
                None
            }
        };

        for seconds in [10usize, 60usize, 300usize] {
            let sample_count = seconds * sampling_rate as usize;
            let samples = (0..sample_count)
                .map(|i| {
                    let t = i as f32 / sampling_rate as f32;
                    0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                        + 0.25 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                        + 0.10 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                        + 0.05 * (2.0 * std::f32::consts::PI * 1760.0 * t).sin()
                })
                .collect::<Vec<_>>();

            let cuda_begin = Instant::now();
            let cuda_out = cuda
                .compute_mel_spectrogram(&samples)
                .expect("cuda mel spectrogram");
            let cuda_elapsed = cuda_begin.elapsed();
            println!(
                "{}s audio: CUDA {:.2} ms ({} frames)",
                seconds,
                cuda_elapsed.as_secs_f64() * 1000.0,
                cuda_out.len()
            );

            if let Some(wgpu) = &wgpu {
                let wgpu_begin = Instant::now();
                match wgpu.compute_mel_spectrogram(&samples) {
                    Ok(wgpu_out) => {
                        let wgpu_elapsed = wgpu_begin.elapsed();
                        assert_eq!(cuda_out.len(), wgpu_out.len(), "frame count mismatch");
                        let ratio = wgpu_elapsed.as_secs_f64()
                            / cuda_elapsed.as_secs_f64().max(f64::EPSILON);
                        println!(
                            "{}s audio: wgpu {:.2} ms, CUDA speed ratio x{:.2}",
                            seconds,
                            wgpu_elapsed.as_secs_f64() * 1000.0,
                            ratio
                        );
                    }
                    Err(err) => {
                        eprintln!("Skipping wgpu run for {}s audio: {err}", seconds);
                    }
                }
            }
        }
    }
}
