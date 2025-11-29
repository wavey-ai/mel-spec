#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

extern "C" __global__ void mel_kernel(const cufftDoubleComplex* fft,
                                      double* out,
                                      const double* filters,
                                      int frames,
                                      int fft_size,
                                      int bins,
                                      int n_mels) {
    int frame_idx = blockIdx.x;
    int mel_idx = blockIdx.y;
    if (frame_idx >= frames || mel_idx >= n_mels) {
        return;
    }

    const cufftDoubleComplex* frame = fft + frame_idx * fft_size;
    const double* filt = filters + mel_idx * bins;

    extern __shared__ double sdata[];
    double sum = 0.0;
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        double re = frame[i].x;
        double im = frame[i].y;
        double mag = re * re + im * im;
        sum += mag * filt[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // block-wide reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double acc = sdata[0];
        if (acc < 1e-10) acc = 1e-10;
        out[mel_idx + frame_idx * n_mels] = log10(acc);
    }
}

extern "C" cudaError_t launch_mel_kernel(const cufftDoubleComplex* fft,
                                         double* out,
                                         const double* filters,
                                         int frames,
                                         int fft_size,
                                         int n_mels,
                                         cudaStream_t stream) {
    if (frames <= 0 || n_mels <= 0) {
        return cudaErrorInvalidValue;
    }
    int bins = (fft_size / 2) + 1;
    int threads = 256;
    size_t shared = threads * sizeof(double);
    dim3 grid(frames, n_mels, 1);
    mel_kernel<<<grid, threads, shared, stream>>>(fft, out, filters, frames, fft_size, bins, n_mels);
    return cudaGetLastError();
}

extern "C" __global__ void window_kernel(const double* hop,
                                         const double* window,
                                         cufftDoubleComplex* out,
                                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double v = hop[i] * window[i];
        out[i].x = v;
        out[i].y = 0.0;
    }
}

extern "C" cudaError_t launch_window_kernel(const double* hop,
                                            const double* window,
                                            cufftDoubleComplex* out,
                                            int n,
                                            cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    window_kernel<<<blocks, threads, 0, stream>>>(hop, window, out, n);
    return cudaGetLastError();
}
