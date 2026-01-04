use ndarray::Array1;
use num::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

#[cfg(all(feature = "cuda", test))]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "cuda")]
use cuda::CudaPlan;

pub struct Spectrogram {
    fft_size: usize,
    idx: u64,
    hop_buf: Vec<f64>,
    hop_size: usize,
    complex_buf: Vec<Complex<f64>>,
    scratch_buf: Vec<Complex<f64>>,
    fft: Arc<dyn Fft<f64>>,
    #[cfg(feature = "cuda")]
    cuda_plan: Option<CudaPlan>,
    #[cfg(feature = "cuda")]
    use_cuda: bool,
    window: Vec<f64>,
}

/// Short Time Fast Fourier Transform
/// Nearly identical to whisper.cpp, pytorch, etc, but the caller might be mindful of the
/// first and final frames:
///   a) pass in exact fft-size sample for initial window to avoid automatic zero-padding
///   b) be aware the final frame will be zero-padded if it is < hop size.
///     - neither is necessary unless you are running additional analysis.
impl Spectrogram {
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        // Hann window
        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - f64::cos((2.0 * PI * i as f64) / fft_size as f64)))
            .collect();
        let idx = 0;

        Self {
            fft_size,
            idx,
            hop_buf: vec![0.0; fft_size],
            hop_size,
            complex_buf: vec![Complex::new(0.0, 0.0); fft_size],
            scratch_buf: vec![Complex::new(0.0, 0.0); fft_size],
            fft,
            #[cfg(feature = "cuda")]
            cuda_plan: None,
            #[cfg(feature = "cuda")]
            use_cuda: false,
            window,
        }
    }

    /// Create a Spectrogram that prefers the CUDA backend.
    ///
    /// Falls back to CPU if CUDA is not available at runtime.
    #[cfg(feature = "cuda")]
    pub fn new_cuda(fft_size: usize, hop_size: usize) -> Result<Self, StftError> {
        let mut s = Self::new(fft_size, hop_size);
        let plan = CudaPlan::new(fft_size)?;
        s.cuda_plan = Some(plan);
        s.use_cuda = true;
        Ok(s)
    }

    /// Process all audio samples at once with batched GPU FFT.
    /// Returns all FFT frames as a single Vec.
    /// This is much faster than calling add() per frame because:
    /// - Single memory transfer to GPU
    /// - Batched cuFFT execution
    /// - Single memory transfer back
    #[cfg(feature = "cuda")]
    pub fn compute_all_cuda(
        samples: &[f32],
        fft_size: usize,
        hop_size: usize,
    ) -> Result<Vec<Vec<Complex<f64>>>, StftError> {
        use std::f64::consts::PI;

        // Calculate number of frames
        if samples.len() < fft_size {
            return Ok(Vec::new());
        }
        let num_frames = (samples.len() - fft_size) / hop_size + 1;
        if num_frames == 0 {
            return Ok(Vec::new());
        }

        // Create Hann window
        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - f64::cos((2.0 * PI * i as f64) / fft_size as f64)))
            .collect();

        // Prepare all windowed frames
        let mut all_windowed: Vec<f64> = Vec::with_capacity(num_frames * fft_size);
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            for i in 0..fft_size {
                let sample = if start + i < samples.len() {
                    samples[start + i] as f64
                } else {
                    0.0
                };
                all_windowed.push(sample * window[i]);
            }
        }

        // Create batched CUDA plan
        let mut plan = CudaPlan::new_batch(fft_size, num_frames)?;

        // Execute batched FFT
        let result = plan.execute_batch(&all_windowed, num_frames)?;

        // Split result into frames
        let frames: Vec<Vec<Complex<f64>>> = result
            .chunks(fft_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(frames)
    }

    /// Process all audio samples at once (CPU version).
    /// Returns all FFT frames. Use this for fair comparison with compute_all_cuda.
    pub fn compute_all_cpu(
        samples: &[f32],
        fft_size: usize,
        hop_size: usize,
    ) -> Vec<Vec<Complex<f64>>> {
        use rustfft::FftPlanner;

        if samples.len() < fft_size {
            return Vec::new();
        }
        let num_frames = (samples.len() - fft_size) / hop_size + 1;
        if num_frames == 0 {
            return Vec::new();
        }

        // Create Hann window
        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - f64::cos((2.0 * PI * i as f64) / fft_size as f64)))
            .collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let mut scratch = vec![Complex::new(0.0, 0.0); fft_size];

        let mut frames_out = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let mut complex_buf: Vec<Complex<f64>> = (0..fft_size)
                .map(|i| {
                    let sample = if start + i < samples.len() {
                        samples[start + i] as f64
                    } else {
                        0.0
                    };
                    Complex::new(sample * window[i], 0.0)
                })
                .collect();

            fft.process_with_scratch(&mut complex_buf, &mut scratch);
            frames_out.push(complex_buf);
        }

        frames_out
    }

    /// Takes a single channel of audio (non-interleaved, mono, f32).
    /// Returns an FFT frame using overlap-and-save and the configured `hop_size`
    pub fn add(&mut self, frames: &[f32]) -> Option<Array1<Complex<f64>>> {
        let fft_size = self.fft_size;
        let hop_size = self.hop_size;

        let mut pcm_data: Vec<f64> = frames.iter().map(|x| *x as f64).collect();
        let pcm_size = pcm_data.len();
        assert!(pcm_size <= hop_size, "frames must be <= hop_size");

        // zero pad
        if pcm_size < hop_size {
            pcm_data.extend_from_slice(&vec![0.0; hop_size - pcm_size]);
        }

        self.hop_buf.copy_within(hop_size.., 0);
        self.hop_buf[(fft_size - hop_size)..].copy_from_slice(&pcm_data);

        self.idx = self.idx.wrapping_add(pcm_size as u64);

        if self.idx >= fft_size as u64 {
            let windowed_samples: Vec<f64> = self
                .hop_buf
                .iter()
                .enumerate()
                .map(|(j, val)| val * self.window[j])
                .collect();

            #[cfg(feature = "cuda")]
            {
                if self.use_cuda {
                    if let Some(plan) = &mut self.cuda_plan {
                        match plan.execute(&windowed_samples) {
                            Ok(out) => return Some(Array1::from_vec(out)),
                            Err(err) => {
                                eprintln!("CUDA FFT failed, falling back to CPU: {err}");
                                self.use_cuda = false;
                            }
                        }
                    }
                }
            }

            self.complex_buf
                .iter_mut()
                .zip(windowed_samples.iter())
                .for_each(|(c, val)| *c = Complex::new(*val, 0.0));

            self.fft
                .process_with_scratch(&mut self.complex_buf, &mut self.scratch_buf);

            Some(Array1::from_vec(self.complex_buf.clone()))
        } else {
            None
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub enum StftError {
    CudaUnavailable(&'static str),
    Cuda(String),
}

#[cfg(feature = "cuda")]
impl std::fmt::Display for StftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StftError::CudaUnavailable(msg) => write!(f, "CUDA unavailable: {msg}"),
            StftError::Cuda(msg) => write!(f, "CUDA error: {msg}"),
        }
    }
}

#[cfg(feature = "cuda")]
impl std::error::Error for StftError {}

#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use super::{Complex, StftError};
    use std::ffi::c_void;
    use std::ptr;

    const CUDA_SUCCESS: i32 = 0;
    const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
    const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

    pub const CUFFT_SUCCESS: i32 = 0;
    pub const CUFFT_FORWARD: i32 = -1;
    pub const CUFFT_Z2Z: i32 = 0x69;

    type CudaError = i32;
    type CufftResult = i32;
    pub type CufftHandle = i32;
    pub type CudaStream = *mut c_void;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(crate) struct CufftDoubleComplex {
        pub x: f64,
        pub y: f64,
    }

    const CUDA_HOST_ALLOC_DEFAULT: u32 = 0;

    extern "C" {
        fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> CudaError;
        fn cudaFree(dev_ptr: *mut c_void) -> CudaError;
        fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> CudaError;
        fn cudaFreeHost(ptr: *mut c_void) -> CudaError;
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: i32,
            stream: CudaStream,
        ) -> CudaError;
        fn cudaMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: i32,
        ) -> CudaError;
        fn cudaStreamCreate(stream: *mut CudaStream) -> CudaError;
        fn cudaStreamDestroy(stream: CudaStream) -> CudaError;
        fn cudaStreamSynchronize(stream: CudaStream) -> CudaError;

        fn cufftPlan1d(plan: *mut CufftHandle, nx: i32, fft_type: i32, batch: i32) -> CufftResult;
        fn cufftDestroy(plan: CufftHandle) -> CufftResult;
        fn cufftSetStream(plan: CufftHandle, stream: CudaStream) -> CufftResult;
        pub fn cufftExecZ2Z(
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
        ) -> CudaError;

        fn launch_window_kernel(
            hop: *const f64,
            window: *const f64,
            out: *mut CufftDoubleComplex,
            n: i32,
            stream: CudaStream,
        ) -> CudaError;
    }

    pub struct CudaPlan {
        pub plan: CufftHandle,
        pub stream: CudaStream,
        pub d_buf: *mut CufftDoubleComplex,
        h_buf: *mut CufftDoubleComplex,
        fft_size: usize,
        batch: usize,
    }

    impl CudaPlan {
        pub fn new(fft_size: usize) -> Result<Self, StftError> {
            Self::new_batch(fft_size, 1)
        }

        pub fn new_batch(fft_size: usize, batch: usize) -> Result<Self, StftError> {
            let mut plan: CufftHandle = 0;
            let res = unsafe {
                cufftPlan1d(
                    &mut plan as *mut CufftHandle,
                    fft_size as i32,
                    CUFFT_Z2Z,
                    batch as i32,
                )
            };
            if res != CUFFT_SUCCESS {
                return Err(StftError::CudaUnavailable("cufftPlan1d failed"));
            }

            let mut stream: CudaStream = ptr::null_mut();
            let err = unsafe { cudaStreamCreate(&mut stream as *mut CudaStream) };
            if err != CUDA_SUCCESS {
                unsafe {
                    let _ = cufftDestroy(plan);
                }
                return Err(StftError::CudaUnavailable("cudaStreamCreate failed"));
            }

            let set_stream = unsafe { cufftSetStream(plan, stream) };
            if set_stream != CUFFT_SUCCESS {
                unsafe {
                    let _ = cudaStreamDestroy(stream);
                    let _ = cufftDestroy(plan);
                }
                return Err(StftError::CudaUnavailable("cufftSetStream failed"));
            }

            let mut d_buf: *mut CufftDoubleComplex = ptr::null_mut();
            let alloc_err = unsafe {
                cudaMalloc(
                    &mut d_buf as *mut *mut CufftDoubleComplex as *mut *mut c_void,
                    fft_size * batch * std::mem::size_of::<CufftDoubleComplex>(),
                )
            };
            if alloc_err != CUDA_SUCCESS {
                unsafe {
                    let _ = cudaStreamDestroy(stream);
                    let _ = cufftDestroy(plan);
                }
                return Err(StftError::CudaUnavailable("cudaMalloc failed"));
            }

            let mut h_buf: *mut CufftDoubleComplex = ptr::null_mut();
            let host_alloc = unsafe {
                cudaHostAlloc(
                    &mut h_buf as *mut *mut CufftDoubleComplex as *mut *mut c_void,
                    fft_size * batch * std::mem::size_of::<CufftDoubleComplex>(),
                    CUDA_HOST_ALLOC_DEFAULT,
                )
            };
            if host_alloc != CUDA_SUCCESS {
                unsafe {
                    let _ = cudaFree(d_buf as *mut c_void);
                    let _ = cudaStreamDestroy(stream);
                    let _ = cufftDestroy(plan);
                }
                return Err(StftError::CudaUnavailable("cudaHostAlloc failed"));
            }

            Ok(Self {
                plan,
                stream,
                d_buf,
                h_buf,
                fft_size,
                batch,
            })
        }

        pub fn execute(&mut self, windowed_samples: &[f64]) -> Result<Vec<Complex<f64>>, StftError> {
            self.execute_batch(windowed_samples, 1)
        }

        pub fn execute_batch(
            &mut self,
            windowed_samples: &[f64],
            frames: usize,
        ) -> Result<Vec<Complex<f64>>, StftError> {
            if frames == 0 || frames > self.batch {
                return Err(StftError::Cuda("invalid batch size".into()));
            }
            if windowed_samples.len() != frames * self.fft_size {
                return Err(StftError::Cuda("input length mismatch".into()));
            }

            let host_slice =
                unsafe { std::slice::from_raw_parts_mut(self.h_buf, self.fft_size * self.batch) };

            // Copy frames into pinned host buffer; zero any unused slots to keep deterministic output.
            for (i, chunk) in windowed_samples.chunks(self.fft_size).enumerate() {
                let offset = i * self.fft_size;
                for (j, &v) in chunk.iter().enumerate() {
                    host_slice[offset + j].x = v;
                    host_slice[offset + j].y = 0.0;
                }
            }
            if frames < self.batch {
                for zero_idx in (frames * self.fft_size)..(self.batch * self.fft_size) {
                    host_slice[zero_idx].x = 0.0;
                    host_slice[zero_idx].y = 0.0;
                }
            }

            let byte_len = self.fft_size * self.batch * std::mem::size_of::<CufftDoubleComplex>();
            let copy_h2d = unsafe {
                cudaMemcpyAsync(
                    self.d_buf as *mut c_void,
                    host_slice.as_ptr() as *const c_void,
                    byte_len,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    self.stream,
                )
            };
            if copy_h2d != CUDA_SUCCESS {
                return Err(StftError::Cuda("cudaMemcpy H2D failed".into()));
            }

            let exec = unsafe { cufftExecZ2Z(self.plan, self.d_buf, self.d_buf, CUFFT_FORWARD) };
            if exec != CUFFT_SUCCESS {
                return Err(StftError::Cuda("cufftExecZ2Z failed".into()));
            }

            let copy_d2h = unsafe {
                cudaMemcpyAsync(
                    host_slice.as_mut_ptr() as *mut c_void,
                    self.d_buf as *const c_void,
                    byte_len,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                    self.stream,
                )
            };
            if copy_d2h != CUDA_SUCCESS {
                return Err(StftError::Cuda("cudaMemcpy D2H failed".into()));
            }

            let sync = unsafe { cudaStreamSynchronize(self.stream) };
            if sync != CUDA_SUCCESS {
                return Err(StftError::Cuda("cudaStreamSynchronize failed".into()));
            }

            let out: Vec<Complex<f64>> = host_slice
                .iter()
                .take(frames * self.fft_size)
                .map(|v| Complex::new(v.x, v.y))
                .collect();

            Ok(out)
        }

        pub fn batch(&self) -> usize {
            self.batch
        }

        pub fn fft_device(
            &mut self,
            windowed_samples: &[f64],
            frames: usize,
        ) -> Result<(), StftError> {
            if frames == 0 || frames > self.batch {
                return Err(StftError::Cuda("invalid batch size".into()));
            }
            if windowed_samples.len() != frames * self.fft_size {
                return Err(StftError::Cuda("input length mismatch".into()));
            }

            let host_slice =
                unsafe { std::slice::from_raw_parts_mut(self.h_buf, self.fft_size * self.batch) };

            for (i, chunk) in windowed_samples.chunks(self.fft_size).enumerate() {
                let offset = i * self.fft_size;
                for (j, &v) in chunk.iter().enumerate() {
                    host_slice[offset + j].x = v;
                    host_slice[offset + j].y = 0.0;
                }
            }
            if frames < self.batch {
                for zero_idx in (frames * self.fft_size)..(self.batch * self.fft_size) {
                    host_slice[zero_idx].x = 0.0;
                    host_slice[zero_idx].y = 0.0;
                }
            }

            let byte_len = self.fft_size * self.batch * std::mem::size_of::<CufftDoubleComplex>();
            let copy_h2d = unsafe {
                cudaMemcpyAsync(
                    self.d_buf as *mut c_void,
                    host_slice.as_ptr() as *const c_void,
                    byte_len,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    self.stream,
                )
            };
            if copy_h2d != CUDA_SUCCESS {
                return Err(StftError::Cuda("cudaMemcpy H2D failed".into()));
            }

            let exec = unsafe { cufftExecZ2Z(self.plan, self.d_buf, self.d_buf, CUFFT_FORWARD) };
            if exec != CUFFT_SUCCESS {
                return Err(StftError::Cuda("cufftExecZ2Z failed".into()));
            }

            let sync = unsafe { cudaStreamSynchronize(self.stream) };
            if sync != CUDA_SUCCESS {
                return Err(StftError::Cuda("cudaStreamSynchronize failed".into()));
            }

            Ok(())
        }

        pub fn device_ptr(&self) -> *const c_void {
            self.d_buf as *const c_void
        }

        pub fn stream(&self) -> CudaStream {
            self.stream
        }
    }

    pub fn alloc_device(size: usize) -> Result<*mut c_void, StftError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr as *mut *mut c_void, size) };
        if err != CUDA_SUCCESS {
            return Err(StftError::Cuda("cudaMalloc failed".into()));
        }
        Ok(ptr)
    }

    pub fn alloc_host(size: usize) -> Result<*mut c_void, StftError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let err = unsafe { cudaHostAlloc(&mut ptr as *mut *mut c_void, size, CUDA_HOST_ALLOC_DEFAULT) };
        if err != CUDA_SUCCESS {
            return Err(StftError::Cuda("cudaHostAlloc failed".into()));
        }
        Ok(ptr)
    }

    pub fn free_device(ptr: *mut c_void) {
        unsafe {
            let _ = cudaFree(ptr);
        }
    }

    pub fn free_host(ptr: *mut c_void) {
        unsafe {
            let _ = cudaFreeHost(ptr);
        }
    }

    pub fn memcpy_h2d_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: CudaStream,
    ) -> Result<(), StftError> {
        let err = unsafe { cudaMemcpyAsync(dst, src, size, CUDA_MEMCPY_HOST_TO_DEVICE, stream) };
        if err != CUDA_SUCCESS {
            return Err(StftError::Cuda("cudaMemcpyAsync H2D failed".into()));
        }
        Ok(())
    }

    pub fn memcpy_d2h_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: CudaStream,
    ) -> Result<(), StftError> {
        let err = unsafe { cudaMemcpyAsync(dst, src, size, CUDA_MEMCPY_DEVICE_TO_HOST, stream) };
        if err != CUDA_SUCCESS {
            return Err(StftError::Cuda("cudaMemcpyAsync D2H failed".into()));
        }
        Ok(())
    }

    pub fn memcpy_d2d_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: CudaStream,
    ) -> Result<(), StftError> {
        let err = unsafe { cudaMemcpyAsync(dst, src, size, CUDA_MEMCPY_DEVICE_TO_DEVICE, stream) };
        if err != CUDA_SUCCESS {
            return Err(StftError::Cuda("cudaMemcpyAsync D2D failed".into()));
        }
        Ok(())
    }

    pub fn stream_sync(stream: CudaStream) -> Result<(), StftError> {
        let err = unsafe { cudaStreamSynchronize(stream) };
        if err != CUDA_SUCCESS {
            return Err(StftError::Cuda("cudaStreamSynchronize failed".into()));
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
    ) -> Result<(), StftError> {
        let err = unsafe {
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
        if err != CUDA_SUCCESS {
            return Err(StftError::Cuda("launch_mel_kernel failed".into()));
        }
        Ok(())
    }

    pub fn launch_window(
        hop: *const f64,
        window: *const f64,
        out: *mut c_void,
        n: usize,
        stream: CudaStream,
    ) -> Result<(), StftError> {
        let err = unsafe { launch_window_kernel(hop, window, out as *mut CufftDoubleComplex, n as i32, stream) };
        if err != CUDA_SUCCESS {
            return Err(StftError::Cuda("launch_window_kernel failed".into()));
        }
        Ok(())
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrogram_add() {
        let fft_size = 8;
        let hop_size = 4;
        let mut spectrogram = Spectrogram::new(fft_size, hop_size);

        // Test with frames that have size less than hop_size
        let frames: Vec<f32> = vec![1.0, 2.0, 3.0];
        let fft_frame = spectrogram.add(&frames);
        assert!(fft_frame.is_none());

        // Test with frames that have size equal to hop_size
        let frames: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let fft_frame = spectrogram.add(&frames);
        // None as we have added 7 frames and fft size is 8
        assert!(fft_frame.is_none());
        let frames: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let fft_frame = spectrogram.add(&frames);
        assert!(fft_frame.is_some());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_matches_cpu() {
        let fft_size = 8;
        let hop_size = 4;

        let mut cpu = Spectrogram::new(fft_size, hop_size);
        let mut gpu = match Spectrogram::new_cuda(fft_size, hop_size) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("skipping CUDA test: {err}");
                return;
            }
        };

        // Prime both with the same data so the first returned frame aligns.
        let frames1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let frames2: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        cpu.add(&frames1);
        gpu.add(&frames1);
        cpu.add(&frames2);
        gpu.add(&frames2);
        let cpu_fft = cpu.add(&frames2).expect("CPU FFT should return frame");
        let gpu_fft = gpu.add(&frames2).expect("CUDA FFT should return frame");

        for (a, b) in cpu_fft.iter().zip(gpu_fft.iter()) {
            assert!((a.re - b.re).abs() < 1e-6);
            assert!((a.im - b.im).abs() < 1e-6);
        }
    }

    #[cfg(feature = "cuda")]
    fn read_wav_samples(path: &str) -> Vec<f32> {
        use soundkit::{audio_bytes::deinterleave_vecs_f32, wav::WavStreamProcessor};
        use std::{fs::File, io::Read};

        let mut file = File::open(path).expect("missing wav fixture");
        let mut processor = WavStreamProcessor::new();
        let mut buf = [0_u8; 4096];
        let mut samples = Vec::new();

        loop {
            let n = file.read(&mut buf).expect("failed to read wav chunk");
            if n == 0 {
                break;
            }

            if let Ok(Some(audio)) = processor.add(&buf[..n]) {
                let chans = deinterleave_vecs_f32(audio.data(), 1);
                samples.extend_from_slice(&chans[0]);
            }
        }

        samples
    }

    #[cfg(feature = "cuda")]
    fn repeat_samples(samples: &[f32], times: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples.len() * times);
        for _ in 0..times {
            out.extend_from_slice(samples);
        }
        out
    }

    #[cfg(feature = "cuda")]
    fn windowed_frames(samples: &[f32], config: &crate::config::MelConfig) -> Vec<Vec<f64>> {
        let fft_size = config.fft_size();
        let hop_size = config.hop_size();
        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - f64::cos((2.0 * PI * i as f64) / fft_size as f64)))
            .collect();

        let mut hop_buf = vec![0.0_f64; fft_size];
        let mut idx: u64 = 0;
        let mut frames = Vec::new();

        for chunk in samples.chunks(hop_size) {
            let mut pcm: Vec<f64> = chunk.iter().map(|x| *x as f64).collect();
            if pcm.len() < hop_size {
                pcm.extend_from_slice(&vec![0.0; hop_size - pcm.len()]);
            }

            hop_buf.copy_within(hop_size.., 0);
            hop_buf[(fft_size - hop_size)..].copy_from_slice(&pcm);
            idx = idx.wrapping_add(chunk.len() as u64);

            if idx >= fft_size as u64 {
                let frame: Vec<f64> = hop_buf
                    .iter()
                    .enumerate()
                    .map(|(j, val)| val * window[j])
                    .collect();
                frames.push(frame);
            }
        }

        frames
    }

    #[cfg(feature = "cuda")]
    fn spawn_nvidia_smi_monitor(
        pid: u32,
    ) -> Option<(Arc<AtomicBool>, std::thread::JoinHandle<()>)> {
        use std::{process::Command, thread, time::Duration};

        let seen = Arc::new(AtomicBool::new(false));
        let flag = seen.clone();

        let handle = thread::Builder::new()
            .name("nvidia-smi-monitor".into())
            .spawn(move || {
                let interval = Duration::from_millis(100);
                for _ in 0..60 {
                    let output = Command::new("nvidia-smi")
                        .args([
                            "--query-compute-apps=pid,process_name,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ])
                        .output();

                    if let Ok(out) = output {
                        if out.status.success() {
                            let text = String::from_utf8_lossy(&out.stdout);
                            for line in text.lines() {
                                if line.split(',').next().map(|s| s.trim())
                                    == Some(pid.to_string().as_str())
                                {
                                    flag.store(true, Ordering::Relaxed);
                                    return;
                                }
                            }
                        }
                    }
                    thread::sleep(interval);
                }
            })
            .ok()?;

        Some((seen, handle))
    }

    #[cfg(feature = "cuda")]
    fn run_jfk_spectrogram(
        samples: &[f32],
        config: &crate::config::MelConfig,
        make_spec: impl FnOnce() -> Result<Spectrogram, StftError>,
    ) -> Result<(Vec<f32>, std::time::Duration, usize, bool), StftError> {
        use crate::mel::{interleave_frames, MelSpectrogram};
        use std::time::Instant;

        let mut spectrogram = make_spec()?;
        let mut mel = MelSpectrogram::new(config.fft_size(), config.sampling_rate(), config.n_mels());
        let mut frames = Vec::new();
        let start = Instant::now();

        for chunk in samples.chunks(config.hop_size()) {
            if let Some(fft) = spectrogram.add(chunk) {
                frames.push(mel.add(&fft));
            }
        }

        let elapsed = start.elapsed();
        let interleaved = interleave_frames(&frames, false, 0);

        Ok((interleaved, elapsed, frames.len(), spectrogram.use_cuda))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_jfk_spectrogram_metrics() {
        use crate::quant::save_tga_8bit;

        let config = crate::config::MelConfig::new(512, 160, 80, 16_000.0);
        let samples = read_wav_samples("./testdata/jfk_f32le.wav");

        // Warm both paths to avoid first-call overhead (CUDA context, FFT plans).
        let _ = run_jfk_spectrogram(&samples, &config, || {
            Ok(Spectrogram::new(config.fft_size(), config.hop_size()))
        });
        let _ = run_jfk_spectrogram(&samples, &config, || {
            Spectrogram::new_cuda(config.fft_size(), config.hop_size())
        });

        let (cpu_interleaved, cpu_elapsed, cpu_frames, _) =
            run_jfk_spectrogram(&samples, &config, || Ok(Spectrogram::new(config.fft_size(), config.hop_size())))
                .expect("CPU spectrogram");

        let smi_monitor = spawn_nvidia_smi_monitor(std::process::id());

        let (gpu_interleaved, gpu_elapsed, gpu_frames, gpu_used_cuda) = {
            // Streaming GPU path: per-frame window/FFT/mel on GPU.
            use crate::stft::cuda::{self, CudaPlan};
            use std::mem::size_of;
            use std::os::raw::c_void;

            let hop_size = config.hop_size();
            let fft_size = config.fft_size();
            let n_mels = config.n_mels();

            let plan = match CudaPlan::new_batch(fft_size, 1) {
                Ok(p) => p,
                Err(err) => {
                    eprintln!("skipping CUDA JFK spectrogram metrics: {err}");
                    return;
                }
            };

            // Host pinned buffers
            let h_mel_out = match cuda::alloc_host(n_mels * size_of::<f64>()) {
                Ok(ptr) => ptr as *mut f64,
                Err(err) => {
                    eprintln!("skipping CUDA JFK spectrogram metrics: {err}");
                    return;
                }
            };
            let d_mel_out = match cuda::alloc_device(n_mels * size_of::<f64>()) {
                Ok(ptr) => ptr,
                Err(err) => {
                    eprintln!("skipping CUDA JFK spectrogram metrics: {err}");
                    return;
                }
            };

            let mel_filters = crate::mel::mel(
                config.sampling_rate(),
                fft_size,
                n_mels,
                None,
                None,
                false,
                true,
            );
            let mel_filters_vec = mel_filters.into_raw_vec();
            let filter_bytes = mel_filters_vec.len() * size_of::<f64>();
            let d_filters = match cuda::alloc_device(filter_bytes) {
                Ok(ptr) => ptr,
                Err(err) => {
                    eprintln!("skipping CUDA JFK spectrogram metrics: {err}");
                    return;
                }
            };
            if let Err(err) = cuda::memcpy_h2d_async(
                d_filters,
                mel_filters_vec.as_ptr() as *const c_void,
                filter_bytes,
                plan.stream(),
            )
            .and_then(|_| cuda::stream_sync(plan.stream()))
            {
                eprintln!("skipping CUDA JFK spectrogram metrics: {err}");
                return;
            }

            let window: Vec<f64> = (0..fft_size)
                .map(|i| 0.5 * (1.0 - f64::cos((2.0 * PI * i as f64) / fft_size as f64)))
                .collect();
            let mut hop_buf = vec![0.0_f64; fft_size];
            let mut frame_buf: Vec<cuda::CufftDoubleComplex> = vec![
                cuda::CufftDoubleComplex { x: 0.0, y: 0.0 };
                fft_size
            ];
            let mut idx: u64 = 0;

            let mut mels = Vec::new();
            let start = std::time::Instant::now();
            for chunk in samples.chunks(hop_size) {
                // Maintain hop buffer on host, window, then send to GPU.
                hop_buf.copy_within(hop_size.., 0);
                hop_buf[(fft_size - hop_size)..].fill(0.0);
                for (i, &v) in chunk.iter().enumerate() {
                    hop_buf[fft_size - hop_size + i] = v as f64;
                }

                for (i, (&s, &w)) in hop_buf.iter().zip(window.iter()).enumerate() {
                    frame_buf[i].x = s * w;
                    frame_buf[i].y = 0.0;
                }

                idx = idx.wrapping_add(chunk.len() as u64);
                if idx < fft_size as u64 {
                    continue;
                }

                let byte_len = fft_size * size_of::<cuda::CufftDoubleComplex>();
                if let Err(err) = cuda::memcpy_h2d_async(
                    plan.d_buf as *mut c_void,
                    frame_buf.as_ptr() as *const c_void,
                    byte_len,
                    plan.stream(),
                ) {
                    eprintln!("CUDA frame H2D failed: {err}");
                    return;
                }

                // FFT in-place
                let exec = unsafe {
                    crate::stft::cuda::cufftExecZ2Z(
                        plan.plan,
                        plan.d_buf,
                        plan.d_buf,
                        crate::stft::cuda::CUFFT_FORWARD,
                    )
                };
                if exec != crate::stft::cuda::CUFFT_SUCCESS {
                    eprintln!("CUDA FFT failed");
                    return;
                }

                // Mel kernel
                if let Err(err) = cuda::launch_mel(
                    plan.device_ptr(),
                    d_mel_out as *mut f64,
                    d_filters as *const f64,
                    1,
                    fft_size,
                    n_mels,
                    plan.stream(),
                ) {
                    eprintln!("CUDA mel launch failed: {err}");
                    return;
                }

                // Copy mel back
                if let Err(err) = cuda::memcpy_d2h_async(
                    h_mel_out as *mut c_void,
                    d_mel_out,
                    n_mels * size_of::<f64>(),
                    plan.stream(),
                )
                .and_then(|_| cuda::stream_sync(plan.stream()))
                {
                    eprintln!("CUDA mel D2H failed: {err}");
                    return;
                }

                let host_slice = unsafe { std::slice::from_raw_parts(h_mel_out, n_mels) };
                let base = ndarray::Array2::from_shape_vec((n_mels, 1), host_slice.to_vec())
                    .expect("mel reshape");
                let normed = crate::mel::norm_mel(&base);
                mels.push(normed);
            }
            let elapsed = start.elapsed();
            let interleaved = crate::mel::interleave_frames(&mels, false, 0);

            cuda::free_device(d_filters);
            cuda::free_device(d_mel_out);
            cuda::free_host(h_mel_out as *mut c_void);

            (interleaved, elapsed, mels.len(), true)
        };

        assert!(
            gpu_used_cuda,
            "CUDA path was not used; fell back to CPU during spectrogram generation"
        );

        assert_eq!(
            cpu_frames, gpu_frames,
            "CUDA and CPU produced different frame counts"
        );
        assert_eq!(
            cpu_interleaved.len(),
            gpu_interleaved.len(),
            "CUDA and CPU produced different output sizes"
        );

        let max_delta = cpu_interleaved
            .iter()
            .zip(gpu_interleaved.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_delta < 1e-4,
            "CUDA spectrogram diverged from CPU (max delta {max_delta})"
        );

        let cpu_path = "./testdata/jfk_spectrogram_cpu.tga";
        let gpu_path = "./testdata/jfk_spectrogram_gpu.tga";
        save_tga_8bit(&cpu_interleaved, config.n_mels(), cpu_path).expect("write CPU image");
        save_tga_8bit(&gpu_interleaved, config.n_mels(), gpu_path).expect("write CUDA image");

        println!(
            "JFK spectrogram: {cpu_frames} frames; CPU {:?}, CUDA {:?}; max delta {:.3e}",
            cpu_elapsed, gpu_elapsed, max_delta
        );

        if let Some((flag, handle)) = smi_monitor {
            let _ = handle.join();
            let saw_pid = flag.load(Ordering::Relaxed);
            assert!(
                saw_pid,
                "nvidia-smi did not report this PID using the GPU during the test"
            );
            println!("nvidia-smi observed this process on the GPU");
        } else {
            eprintln!("nvidia-smi not available; skipping GPU process observation");
        }
        if gpu_elapsed.as_nanos() > 0 {
            let speedup = cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64().max(f64::EPSILON);
            println!(
                "Images written to {cpu_path} and {gpu_path}; CUDA speedup x{speedup:.2}"
            );
        } else {
            println!("Images written to {cpu_path} and {gpu_path}");
        }
    }
}
