use crate::stft::{frame_windows, hann_window};
use ::wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use rustfft::{num_complex::Complex32 as FftComplex32, FftPlanner};
use std::fmt;
use std::sync::mpsc;

const SHADER: &str = include_str!("wgpu.wgsl");
const MAX_DISPATCH_GROUPS: u32 = 65_535;
const WORKGROUP_SIZE: u32 = 64;

#[derive(Debug)]
pub enum WgpuError {
    AdapterUnavailable,
    BufferMap(String),
    DevicePoll(String),
    RequestAdapter(String),
    RequestDevice(String),
    UnsupportedFftSize(usize),
}

impl fmt::Display for WgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterUnavailable => write!(f, "no suitable GPU adapter was found"),
            Self::BufferMap(msg) => write!(f, "failed to map GPU buffer: {msg}"),
            Self::DevicePoll(msg) => write!(f, "failed while waiting for GPU work: {msg}"),
            Self::RequestAdapter(msg) => write!(f, "failed to request GPU adapter: {msg}"),
            Self::RequestDevice(msg) => write!(f, "failed to request GPU device: {msg}"),
            Self::UnsupportedFftSize(size) => {
                write!(f, "wgpu backend does not support fft_size={size}")
            }
        }
    }
}

impl std::error::Error for WgpuError {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Complex32 {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BitReverseUniforms {
    fft_size: u32,
    num_frames: u32,
    log2_size: u32,
    dispatch_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StageUniforms {
    fft_size: u32,
    num_frames: u32,
    stage_len: u32,
    half_len: u32,
    dispatch_offset: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MelUniforms {
    fft_size: u32,
    num_frames: u32,
    n_mels: u32,
    bins: u32,
    log10_scale: f32,
    epsilon: f32,
    dispatch_offset: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BluesteinPrepareUniforms {
    fft_size: u32,
    convolution_size: u32,
    num_frames: u32,
    dispatch_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointwiseUniforms {
    fft_size: u32,
    num_frames: u32,
    dispatch_offset: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ConjugateUniforms {
    total_len: u32,
    dispatch_offset: u32,
    scale: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BluesteinPostUniforms {
    fft_size: u32,
    convolution_size: u32,
    num_frames: u32,
    dispatch_offset: u32,
    inverse_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct WgpuMelSpectrogram {
    adapter_info: ::wgpu::AdapterInfo,
    bitreverse_bgl: ::wgpu::BindGroupLayout,
    bitreverse_pipeline: ::wgpu::ComputePipeline,
    bluestein_convolution_size: Option<usize>,
    bluestein_kernel_buffer: Option<::wgpu::Buffer>,
    bluestein_post_pipeline: ::wgpu::ComputePipeline,
    bluestein_prepare_pipeline: ::wgpu::ComputePipeline,
    conjugate_pipeline: ::wgpu::ComputePipeline,
    device: ::wgpu::Device,
    fft_size: usize,
    filter_buffer: ::wgpu::Buffer,
    hop_size: usize,
    mel_bgl: ::wgpu::BindGroupLayout,
    mel_pipeline: ::wgpu::ComputePipeline,
    n_mels: usize,
    pointwise_bgl: ::wgpu::BindGroupLayout,
    pointwise_pipeline: ::wgpu::ComputePipeline,
    queue: ::wgpu::Queue,
    stage_bgl: ::wgpu::BindGroupLayout,
    stage_pipeline: ::wgpu::ComputePipeline,
}

impl WgpuMelSpectrogram {
    pub fn new(
        fft_size: usize,
        hop_size: usize,
        sampling_rate: f64,
        n_mels: usize,
    ) -> Result<Self, WgpuError> {
        if fft_size == 0 {
            return Err(WgpuError::UnsupportedFftSize(fft_size));
        }

        let instance = ::wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&::wgpu::RequestAdapterOptions {
                power_preference: ::wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            }))
            .map_err(|err| WgpuError::RequestAdapter(err.to_string()))?;

        let adapter_info = adapter.get_info();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&::wgpu::DeviceDescriptor {
                label: Some("mel-spec wgpu device"),
                required_features: ::wgpu::Features::empty(),
                required_limits: ::wgpu::Limits::default(),
                memory_hints: ::wgpu::MemoryHints::Performance,
                trace: ::wgpu::Trace::Off,
                experimental_features: Default::default(),
            }))
            .map_err(|err| WgpuError::RequestDevice(err.to_string()))?;

        let shader = device.create_shader_module(::wgpu::ShaderModuleDescriptor {
            label: Some("mel-spec wgpu shader"),
            source: ::wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bitreverse_bgl = device.create_bind_group_layout(&::wgpu::BindGroupLayoutDescriptor {
            label: Some("mel-spec bitreverse bgl"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, false),
                uniform_layout_entry(2),
            ],
        });

        let stage_bgl = device.create_bind_group_layout(&::wgpu::BindGroupLayoutDescriptor {
            label: Some("mel-spec fft stage bgl"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, false),
                uniform_layout_entry(2),
            ],
        });

        let pointwise_bgl = device.create_bind_group_layout(&::wgpu::BindGroupLayoutDescriptor {
            label: Some("mel-spec pointwise bgl"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, false),
                uniform_layout_entry(3),
            ],
        });

        let mel_bgl = device.create_bind_group_layout(&::wgpu::BindGroupLayoutDescriptor {
            label: Some("mel-spec mel bgl"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, false),
                uniform_layout_entry(3),
            ],
        });

        let bitreverse_pipeline = create_pipeline(
            &device,
            &shader,
            &bitreverse_bgl,
            "bitreverse_main",
            "bitreverse",
        );
        let stage_pipeline =
            create_pipeline(&device, &shader, &stage_bgl, "fft_stage_main", "fft stage");
        let bluestein_prepare_pipeline = create_pipeline(
            &device,
            &shader,
            &stage_bgl,
            "bluestein_prepare_main",
            "bluestein prepare",
        );
        let conjugate_pipeline =
            create_pipeline(&device, &shader, &stage_bgl, "conjugate_main", "conjugate");
        let bluestein_post_pipeline = create_pipeline(
            &device,
            &shader,
            &stage_bgl,
            "bluestein_post_main",
            "bluestein post",
        );
        let pointwise_pipeline = create_pipeline(
            &device,
            &shader,
            &pointwise_bgl,
            "pointwise_main",
            "pointwise multiply",
        );
        let mel_pipeline = create_pipeline(&device, &shader, &mel_bgl, "mel_main", "mel");

        let filters = crate::mel::mel(sampling_rate, fft_size, n_mels, None, None, false, true)
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();

        let filter_buffer = device.create_buffer_init(&::wgpu::util::BufferInitDescriptor {
            label: Some("mel-spec filter buffer"),
            contents: bytemuck::cast_slice(&filters),
            usage: ::wgpu::BufferUsages::STORAGE,
        });

        let (bluestein_convolution_size, bluestein_kernel_buffer) = if fft_size.is_power_of_two() {
            (None, None)
        } else {
            let (convolution_size, kernel_fft) = build_bluestein_kernel(fft_size);
            let kernel_buffer = device.create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                label: Some("mel-spec bluestein kernel"),
                contents: bytemuck::cast_slice(&kernel_fft),
                usage: ::wgpu::BufferUsages::STORAGE,
            });
            (Some(convolution_size), Some(kernel_buffer))
        };

        Ok(Self {
            adapter_info,
            bitreverse_bgl,
            bitreverse_pipeline,
            bluestein_convolution_size,
            bluestein_kernel_buffer,
            bluestein_post_pipeline,
            bluestein_prepare_pipeline,
            conjugate_pipeline,
            device,
            fft_size,
            filter_buffer,
            hop_size,
            mel_bgl,
            mel_pipeline,
            n_mels,
            pointwise_bgl,
            pointwise_pipeline,
            queue,
            stage_bgl,
            stage_pipeline,
        })
    }

    pub fn adapter_info(&self) -> &::wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub fn compute_mel_spectrogram(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>, WgpuError> {
        let window = hann_window(self.fft_size);
        let frames = frame_windows(samples, self.fft_size, self.hop_size, &window);
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let mut output = Vec::with_capacity(frames.len());
        for batch in frames.chunks(self.max_frames_per_batch()) {
            output.extend(self.compute_batch_mel_spectrogram(batch)?);
        }

        Ok(output)
    }

    fn compute_batch_mel_spectrogram(
        &self,
        frames: &[Vec<f64>],
    ) -> Result<Vec<Vec<f32>>, WgpuError> {
        let num_frames = frames.len();
        let complex_frames = frames
            .iter()
            .flat_map(|frame| {
                frame.iter().map(|sample| Complex32 {
                    re: *sample as f32,
                    im: 0.0,
                })
            })
            .collect::<Vec<_>>();

        let complex_bytes = size_bytes::<Complex32>(complex_frames.len());
        let mel_values_len = num_frames * self.n_mels;
        let mel_bytes = size_bytes::<f32>(mel_values_len);

        let input_buffer = self
            .device
            .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                label: Some("mel-spec fft input"),
                contents: bytemuck::cast_slice(&complex_frames),
                usage: ::wgpu::BufferUsages::STORAGE,
            });

        let mel_buffer = self.device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some("mel-spec mel buffer"),
            size: mel_bytes,
            usage: ::wgpu::BufferUsages::STORAGE | ::wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = self.device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some("mel-spec mel readback"),
            size: mel_bytes,
            usage: ::wgpu::BufferUsages::COPY_DST | ::wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&::wgpu::CommandEncoderDescriptor {
                label: Some("mel-spec wgpu encoder"),
            });

        let max_invocations_per_dispatch = MAX_DISPATCH_GROUPS * WORKGROUP_SIZE;
        let fft_buffer = if self.fft_size.is_power_of_two() {
            let scratch_buffer = create_storage_buffer(
                &self.device,
                "mel-spec fft scratch",
                complex_bytes,
                ::wgpu::BufferUsages::empty(),
            );
            self.encode_radix2_fft(
                &mut encoder,
                &input_buffer,
                &scratch_buffer,
                num_frames,
                self.fft_size,
                max_invocations_per_dispatch,
            )
        } else {
            let output_buffer = create_storage_buffer(
                &self.device,
                "mel-spec bluestein output",
                complex_bytes,
                ::wgpu::BufferUsages::empty(),
            );
            self.encode_bluestein_fft(
                &mut encoder,
                &input_buffer,
                &output_buffer,
                num_frames,
                max_invocations_per_dispatch,
            );
            output_buffer
        };

        let mel_total = mel_values_len as u32;
        for dispatch_offset in (0..mel_total).step_by(max_invocations_per_dispatch as usize) {
            let chunk_invocations = (mel_total - dispatch_offset).min(max_invocations_per_dispatch);
            let mel_uniforms = MelUniforms {
                fft_size: self.fft_size as u32,
                num_frames: num_frames as u32,
                n_mels: self.n_mels as u32,
                bins: (self.fft_size / 2 + 1) as u32,
                log10_scale: std::f32::consts::LOG10_E,
                epsilon: 1.0e-10,
                dispatch_offset,
                _pad: 0,
            };
            let mel_uniform_buffer =
                self.device
                    .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                        label: Some("mel-spec mel uniforms"),
                        contents: bytemuck::bytes_of(&mel_uniforms),
                        usage: ::wgpu::BufferUsages::UNIFORM,
                    });
            let mel_bind_group = self.device.create_bind_group(&::wgpu::BindGroupDescriptor {
                label: Some("mel-spec mel bind group"),
                layout: &self.mel_bgl,
                entries: &[
                    ::wgpu::BindGroupEntry {
                        binding: 0,
                        resource: fft_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.filter_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 2,
                        resource: mel_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 3,
                        resource: mel_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            dispatch_compute(
                &mut encoder,
                &self.mel_pipeline,
                &mel_bind_group,
                dispatch_count(chunk_invocations),
            );
        }

        encoder.copy_buffer_to_buffer(&mel_buffer, 0, &readback_buffer, 0, mel_bytes);

        let submission = self.queue.submit(Some(encoder.finish()));
        let slice = readback_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(::wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device
            .poll(::wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|err| WgpuError::DevicePoll(err.to_string()))?;

        rx.recv()
            .map_err(|err| WgpuError::BufferMap(err.to_string()))?
            .map_err(|err| WgpuError::BufferMap(err.to_string()))?;

        let view = slice.get_mapped_range();
        let mel_values = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        readback_buffer.unmap();

        let mut output = Vec::with_capacity(num_frames);
        for frame in mel_values.chunks(self.n_mels) {
            output.push(crate::mel::norm_mel_vec(frame));
        }

        Ok(output)
    }

    fn max_frames_per_batch(&self) -> usize {
        let max_storage_binding_size =
            ::wgpu::Limits::default().max_storage_buffer_binding_size as u64;
        let complex_fft_size = if self.fft_size.is_power_of_two() {
            self.fft_size
        } else {
            self.bluestein_convolution_size
                .expect("bluestein size is only needed for non-power-of-two FFTs")
        };
        let complex_frame_bytes = size_bytes::<Complex32>(complex_fft_size);
        let mel_frame_bytes = size_bytes::<f32>(self.n_mels);
        let max_fft_frames = (max_storage_binding_size / complex_frame_bytes).max(1) as usize;
        let max_mel_frames = (max_storage_binding_size / mel_frame_bytes).max(1) as usize;
        max_fft_frames.min(max_mel_frames).max(1)
    }

    fn encode_radix2_fft(
        &self,
        encoder: &mut ::wgpu::CommandEncoder,
        input_buffer: &::wgpu::Buffer,
        scratch_buffer: &::wgpu::Buffer,
        num_frames: usize,
        fft_size: usize,
        max_invocations_per_dispatch: u32,
    ) -> ::wgpu::Buffer {
        let bitreverse_total = (num_frames * fft_size) as u32;
        for dispatch_offset in (0..bitreverse_total).step_by(max_invocations_per_dispatch as usize)
        {
            let chunk_invocations =
                (bitreverse_total - dispatch_offset).min(max_invocations_per_dispatch);
            let bitreverse_uniforms = BitReverseUniforms {
                fft_size: fft_size as u32,
                num_frames: num_frames as u32,
                log2_size: fft_size.ilog2(),
                dispatch_offset,
            };
            let bitreverse_uniform_buffer =
                self.device
                    .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                        label: Some("mel-spec bitreverse uniforms"),
                        contents: bytemuck::bytes_of(&bitreverse_uniforms),
                        usage: ::wgpu::BufferUsages::UNIFORM,
                    });
            let bitreverse_bind_group =
                self.device.create_bind_group(&::wgpu::BindGroupDescriptor {
                    label: Some("mel-spec bitreverse bind group"),
                    layout: &self.bitreverse_bgl,
                    entries: &[
                        ::wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buffer.as_entire_binding(),
                        },
                        ::wgpu::BindGroupEntry {
                            binding: 1,
                            resource: scratch_buffer.as_entire_binding(),
                        },
                        ::wgpu::BindGroupEntry {
                            binding: 2,
                            resource: bitreverse_uniform_buffer.as_entire_binding(),
                        },
                    ],
                });

            dispatch_compute(
                encoder,
                &self.bitreverse_pipeline,
                &bitreverse_bind_group,
                dispatch_count(chunk_invocations),
            );
        }

        let mut read_from_scratch = true;
        let mut stage_len = 2usize;
        while stage_len <= fft_size {
            let half_len = stage_len / 2;
            let (src_buffer, dst_buffer) = if read_from_scratch {
                (scratch_buffer, input_buffer)
            } else {
                (input_buffer, scratch_buffer)
            };

            let stage_total = (num_frames * fft_size / 2) as u32;
            for dispatch_offset in (0..stage_total).step_by(max_invocations_per_dispatch as usize) {
                let chunk_invocations =
                    (stage_total - dispatch_offset).min(max_invocations_per_dispatch);
                let stage_uniforms = StageUniforms {
                    fft_size: fft_size as u32,
                    num_frames: num_frames as u32,
                    stage_len: stage_len as u32,
                    half_len: half_len as u32,
                    dispatch_offset,
                    _pad: 0,
                };
                let stage_uniform_buffer =
                    self.device
                        .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                            label: Some("mel-spec stage uniforms"),
                            contents: bytemuck::bytes_of(&stage_uniforms),
                            usage: ::wgpu::BufferUsages::UNIFORM,
                        });

                let stage_bind_group =
                    self.device.create_bind_group(&::wgpu::BindGroupDescriptor {
                        label: Some("mel-spec stage bind group"),
                        layout: &self.stage_bgl,
                        entries: &[
                            ::wgpu::BindGroupEntry {
                                binding: 0,
                                resource: src_buffer.as_entire_binding(),
                            },
                            ::wgpu::BindGroupEntry {
                                binding: 1,
                                resource: dst_buffer.as_entire_binding(),
                            },
                            ::wgpu::BindGroupEntry {
                                binding: 2,
                                resource: stage_uniform_buffer.as_entire_binding(),
                            },
                        ],
                    });

                dispatch_compute(
                    encoder,
                    &self.stage_pipeline,
                    &stage_bind_group,
                    dispatch_count(chunk_invocations),
                );
            }

            read_from_scratch = !read_from_scratch;
            stage_len *= 2;
        }

        if read_from_scratch {
            scratch_buffer.clone()
        } else {
            input_buffer.clone()
        }
    }

    fn encode_bluestein_fft(
        &self,
        encoder: &mut ::wgpu::CommandEncoder,
        input_buffer: &::wgpu::Buffer,
        output_buffer: &::wgpu::Buffer,
        num_frames: usize,
        max_invocations_per_dispatch: u32,
    ) {
        let convolution_size = self
            .bluestein_convolution_size
            .expect("bluestein size is only needed for non-power-of-two FFTs");
        let kernel_buffer = self
            .bluestein_kernel_buffer
            .as_ref()
            .expect("bluestein kernel buffer is only needed for non-power-of-two FFTs");
        let convolution_bytes = size_bytes::<Complex32>(num_frames * convolution_size);

        let prepared_buffer = create_storage_buffer(
            &self.device,
            "mel-spec bluestein prepared",
            convolution_bytes,
            ::wgpu::BufferUsages::empty(),
        );
        let scratch_buffer = create_storage_buffer(
            &self.device,
            "mel-spec bluestein scratch",
            convolution_bytes,
            ::wgpu::BufferUsages::empty(),
        );
        let pointwise_buffer = create_storage_buffer(
            &self.device,
            "mel-spec bluestein pointwise",
            convolution_bytes,
            ::wgpu::BufferUsages::empty(),
        );

        self.encode_bluestein_prepare(
            encoder,
            input_buffer,
            &prepared_buffer,
            num_frames,
            convolution_size,
            max_invocations_per_dispatch,
        );

        let prepared_fft = self.encode_radix2_fft(
            encoder,
            &prepared_buffer,
            &scratch_buffer,
            num_frames,
            convolution_size,
            max_invocations_per_dispatch,
        );

        self.encode_pointwise_multiply(
            encoder,
            &prepared_fft,
            kernel_buffer,
            &pointwise_buffer,
            num_frames,
            convolution_size,
            max_invocations_per_dispatch,
        );

        self.encode_conjugate(
            encoder,
            &pointwise_buffer,
            &prepared_buffer,
            num_frames * convolution_size,
            max_invocations_per_dispatch,
            1.0,
        );

        let inverse_fft_ready = self.encode_radix2_fft(
            encoder,
            &prepared_buffer,
            &scratch_buffer,
            num_frames,
            convolution_size,
            max_invocations_per_dispatch,
        );

        self.encode_bluestein_postprocess(
            encoder,
            &inverse_fft_ready,
            output_buffer,
            num_frames,
            convolution_size,
            max_invocations_per_dispatch,
        );
    }

    fn encode_bluestein_prepare(
        &self,
        encoder: &mut ::wgpu::CommandEncoder,
        input_buffer: &::wgpu::Buffer,
        output_buffer: &::wgpu::Buffer,
        num_frames: usize,
        convolution_size: usize,
        max_invocations_per_dispatch: u32,
    ) {
        let total = (num_frames * convolution_size) as u32;
        for dispatch_offset in (0..total).step_by(max_invocations_per_dispatch as usize) {
            let chunk_invocations = (total - dispatch_offset).min(max_invocations_per_dispatch);
            let uniforms = BluesteinPrepareUniforms {
                fft_size: self.fft_size as u32,
                convolution_size: convolution_size as u32,
                num_frames: num_frames as u32,
                dispatch_offset,
            };
            let uniform_buffer =
                self.device
                    .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                        label: Some("mel-spec bluestein prepare uniforms"),
                        contents: bytemuck::bytes_of(&uniforms),
                        usage: ::wgpu::BufferUsages::UNIFORM,
                    });
            let bind_group = self.device.create_bind_group(&::wgpu::BindGroupDescriptor {
                label: Some("mel-spec bluestein prepare bind group"),
                layout: &self.stage_bgl,
                entries: &[
                    ::wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            dispatch_compute(
                encoder,
                &self.bluestein_prepare_pipeline,
                &bind_group,
                dispatch_count(chunk_invocations),
            );
        }
    }

    fn encode_pointwise_multiply(
        &self,
        encoder: &mut ::wgpu::CommandEncoder,
        left_buffer: &::wgpu::Buffer,
        right_buffer: &::wgpu::Buffer,
        output_buffer: &::wgpu::Buffer,
        num_frames: usize,
        fft_size: usize,
        max_invocations_per_dispatch: u32,
    ) {
        let total = (num_frames * fft_size) as u32;
        for dispatch_offset in (0..total).step_by(max_invocations_per_dispatch as usize) {
            let chunk_invocations = (total - dispatch_offset).min(max_invocations_per_dispatch);
            let uniforms = PointwiseUniforms {
                fft_size: fft_size as u32,
                num_frames: num_frames as u32,
                dispatch_offset,
                _pad: 0,
            };
            let uniform_buffer =
                self.device
                    .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                        label: Some("mel-spec pointwise uniforms"),
                        contents: bytemuck::bytes_of(&uniforms),
                        usage: ::wgpu::BufferUsages::UNIFORM,
                    });
            let bind_group = self.device.create_bind_group(&::wgpu::BindGroupDescriptor {
                label: Some("mel-spec pointwise bind group"),
                layout: &self.pointwise_bgl,
                entries: &[
                    ::wgpu::BindGroupEntry {
                        binding: 0,
                        resource: left_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 1,
                        resource: right_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            dispatch_compute(
                encoder,
                &self.pointwise_pipeline,
                &bind_group,
                dispatch_count(chunk_invocations),
            );
        }
    }

    fn encode_conjugate(
        &self,
        encoder: &mut ::wgpu::CommandEncoder,
        input_buffer: &::wgpu::Buffer,
        output_buffer: &::wgpu::Buffer,
        total_len: usize,
        max_invocations_per_dispatch: u32,
        scale: f32,
    ) {
        let total = total_len as u32;
        for dispatch_offset in (0..total).step_by(max_invocations_per_dispatch as usize) {
            let chunk_invocations = (total - dispatch_offset).min(max_invocations_per_dispatch);
            let uniforms = ConjugateUniforms {
                total_len: total,
                dispatch_offset,
                scale,
                _pad: 0,
            };
            let uniform_buffer =
                self.device
                    .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                        label: Some("mel-spec conjugate uniforms"),
                        contents: bytemuck::bytes_of(&uniforms),
                        usage: ::wgpu::BufferUsages::UNIFORM,
                    });
            let bind_group = self.device.create_bind_group(&::wgpu::BindGroupDescriptor {
                label: Some("mel-spec conjugate bind group"),
                layout: &self.stage_bgl,
                entries: &[
                    ::wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            dispatch_compute(
                encoder,
                &self.conjugate_pipeline,
                &bind_group,
                dispatch_count(chunk_invocations),
            );
        }
    }

    fn encode_bluestein_postprocess(
        &self,
        encoder: &mut ::wgpu::CommandEncoder,
        input_buffer: &::wgpu::Buffer,
        output_buffer: &::wgpu::Buffer,
        num_frames: usize,
        convolution_size: usize,
        max_invocations_per_dispatch: u32,
    ) {
        let total = (num_frames * self.fft_size) as u32;
        for dispatch_offset in (0..total).step_by(max_invocations_per_dispatch as usize) {
            let chunk_invocations = (total - dispatch_offset).min(max_invocations_per_dispatch);
            let uniforms = BluesteinPostUniforms {
                fft_size: self.fft_size as u32,
                convolution_size: convolution_size as u32,
                num_frames: num_frames as u32,
                dispatch_offset,
                inverse_scale: 1.0 / convolution_size as f32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let uniform_buffer =
                self.device
                    .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                        label: Some("mel-spec bluestein post uniforms"),
                        contents: bytemuck::bytes_of(&uniforms),
                        usage: ::wgpu::BufferUsages::UNIFORM,
                    });
            let bind_group = self.device.create_bind_group(&::wgpu::BindGroupDescriptor {
                label: Some("mel-spec bluestein post bind group"),
                layout: &self.stage_bgl,
                entries: &[
                    ::wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            dispatch_compute(
                encoder,
                &self.bluestein_post_pipeline,
                &bind_group,
                dispatch_count(chunk_invocations),
            );
        }
    }
}

fn build_bluestein_kernel(fft_size: usize) -> (usize, Vec<Complex32>) {
    let convolution_size = (fft_size.saturating_mul(2).saturating_sub(1)).next_power_of_two();
    let mut kernel = vec![FftComplex32::new(0.0, 0.0); convolution_size];

    for n in 0..fft_size {
        let n = n as f32;
        let angle = std::f32::consts::PI * n * n / fft_size as f32;
        let value = FftComplex32::new(angle.cos(), angle.sin());
        kernel[n as usize] = value;
        if n != 0.0 {
            kernel[convolution_size - n as usize] = value;
        }
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(convolution_size);
    fft.process(&mut kernel);

    (
        convolution_size,
        kernel
            .into_iter()
            .map(|value| Complex32 {
                re: value.re,
                im: value.im,
            })
            .collect(),
    )
}

fn create_pipeline(
    device: &::wgpu::Device,
    shader: &::wgpu::ShaderModule,
    bind_group_layout: &::wgpu::BindGroupLayout,
    entry_point: &'static str,
    label: &'static str,
) -> ::wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&::wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });

    device.create_compute_pipeline(&::wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: ::wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn create_storage_buffer(
    device: &::wgpu::Device,
    label: &'static str,
    size: u64,
    extra_usage: ::wgpu::BufferUsages,
) -> ::wgpu::Buffer {
    device.create_buffer(&::wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: ::wgpu::BufferUsages::STORAGE | extra_usage,
        mapped_at_creation: false,
    })
}

fn storage_layout_entry(binding: u32, read_only: bool) -> ::wgpu::BindGroupLayoutEntry {
    ::wgpu::BindGroupLayoutEntry {
        binding,
        visibility: ::wgpu::ShaderStages::COMPUTE,
        ty: ::wgpu::BindingType::Buffer {
            ty: ::wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32) -> ::wgpu::BindGroupLayoutEntry {
    ::wgpu::BindGroupLayoutEntry {
        binding,
        visibility: ::wgpu::ShaderStages::COMPUTE,
        ty: ::wgpu::BindingType::Buffer {
            ty: ::wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn dispatch_compute(
    encoder: &mut ::wgpu::CommandEncoder,
    pipeline: &::wgpu::ComputePipeline,
    bind_group: &::wgpu::BindGroup,
    x_groups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&::wgpu::ComputePassDescriptor {
        label: Some("mel-spec compute pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(x_groups, 1, 1);
}

fn dispatch_count(total_invocations: u32) -> u32 {
    total_invocations.div_ceil(WORKGROUP_SIZE)
}

fn size_bytes<T>(len: usize) -> u64 {
    (std::mem::size_of::<T>() * len) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn wgpu_matches_cpu_for_power_of_two_fft() {
        let fft_size = 512;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16_000.0;
        let samples = (0..16_384)
            .map(|i| {
                let t = i as f32 / sampling_rate as f32;
                0.7 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 1760.0 * t).sin()
            })
            .collect::<Vec<_>>();

        let cpu = crate::stft::Spectrogram::compute_mel_spectrogram_cpu(
            &samples,
            fft_size,
            hop_size,
            n_mels,
            sampling_rate,
        );

        let gpu = match WgpuMelSpectrogram::new(fft_size, hop_size, sampling_rate, n_mels) {
            Ok(gpu) => gpu,
            Err(err) => {
                eprintln!("Skipping wgpu test: {err}");
                return;
            }
        };

        let gpu_out = gpu
            .compute_mel_spectrogram(&samples)
            .expect("wgpu mel spectrogram");

        assert_eq!(cpu.len(), gpu_out.len(), "frame count mismatch");

        let mut max_delta = 0.0f32;
        let mut sum_delta = 0.0f32;
        let mut count = 0usize;
        for (cpu_frame, gpu_frame) in cpu.iter().zip(gpu_out.iter()) {
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
    fn wgpu_matches_cpu_for_whisper_fft_400() {
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

        let cpu = crate::stft::Spectrogram::compute_mel_spectrogram_cpu(
            &samples,
            fft_size,
            hop_size,
            n_mels,
            sampling_rate,
        );

        let gpu = match WgpuMelSpectrogram::new(fft_size, hop_size, sampling_rate, n_mels) {
            Ok(gpu) => gpu,
            Err(err) => {
                eprintln!("Skipping wgpu 400 test: {err}");
                return;
            }
        };

        let gpu_out = gpu
            .compute_mel_spectrogram(&samples)
            .expect("wgpu mel spectrogram");

        assert_eq!(cpu.len(), gpu_out.len(), "frame count mismatch");

        let mut max_delta = 0.0f32;
        let mut sum_delta = 0.0f32;
        let mut count = 0usize;
        for (cpu_frame, gpu_frame) in cpu.iter().zip(gpu_out.iter()) {
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
    fn benchmark_wgpu_vs_cpu() {
        let fft_size = 512;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16_000.0;

        let startup_begin = Instant::now();
        let gpu = match WgpuMelSpectrogram::new(fft_size, hop_size, sampling_rate, n_mels) {
            Ok(gpu) => gpu,
            Err(err) => {
                eprintln!("Skipping wgpu benchmark: {err}");
                return;
            }
        };
        let startup_elapsed = startup_begin.elapsed();
        let adapter = gpu.adapter_info();
        println!(
            "GPU adapter: {} {} ({:?})",
            adapter.vendor, adapter.name, adapter.backend
        );
        println!(
            "GPU startup: {:.2} ms",
            startup_elapsed.as_secs_f64() * 1000.0
        );

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

            let warmup = gpu
                .compute_mel_spectrogram(&samples)
                .expect("wgpu warmup mel spectrogram");
            assert!(!warmup.is_empty(), "expected non-empty warmup output");

            let cpu_begin = Instant::now();
            let cpu = crate::stft::Spectrogram::compute_mel_spectrogram_cpu(
                &samples,
                fft_size,
                hop_size,
                n_mels,
                sampling_rate,
            );
            let cpu_elapsed = cpu_begin.elapsed();

            let gpu_begin = Instant::now();
            let gpu_out = gpu
                .compute_mel_spectrogram(&samples)
                .expect("wgpu mel spectrogram");
            let gpu_elapsed = gpu_begin.elapsed();

            assert_eq!(cpu.len(), gpu_out.len(), "frame count mismatch");

            let speedup = cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64().max(f64::EPSILON);
            println!(
                "{}s audio: CPU {:.2} ms, GPU {:.2} ms, speedup x{:.2}",
                seconds,
                cpu_elapsed.as_secs_f64() * 1000.0,
                gpu_elapsed.as_secs_f64() * 1000.0,
                speedup
            );
        }
    }

    #[test]
    #[ignore]
    fn benchmark_wgpu_vs_cpu_whisper_fft_400() {
        let fft_size = 400;
        let hop_size = 160;
        let n_mels = 80;
        let sampling_rate = 16_000.0;

        let startup_begin = Instant::now();
        let gpu = match WgpuMelSpectrogram::new(fft_size, hop_size, sampling_rate, n_mels) {
            Ok(gpu) => gpu,
            Err(err) => {
                eprintln!("Skipping wgpu benchmark: {err}");
                return;
            }
        };
        let startup_elapsed = startup_begin.elapsed();
        let adapter = gpu.adapter_info();
        println!(
            "GPU adapter: {} {} ({:?})",
            adapter.vendor, adapter.name, adapter.backend
        );
        println!(
            "GPU startup: {:.2} ms",
            startup_elapsed.as_secs_f64() * 1000.0
        );

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

            let warmup = gpu
                .compute_mel_spectrogram(&samples)
                .expect("wgpu warmup mel spectrogram");
            assert!(!warmup.is_empty(), "expected non-empty warmup output");

            let cpu_begin = Instant::now();
            let cpu = crate::stft::Spectrogram::compute_mel_spectrogram_cpu(
                &samples,
                fft_size,
                hop_size,
                n_mels,
                sampling_rate,
            );
            let cpu_elapsed = cpu_begin.elapsed();

            let gpu_begin = Instant::now();
            let gpu_out = gpu
                .compute_mel_spectrogram(&samples)
                .expect("wgpu mel spectrogram");
            let gpu_elapsed = gpu_begin.elapsed();

            assert_eq!(cpu.len(), gpu_out.len(), "frame count mismatch");

            let speedup = cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64().max(f64::EPSILON);
            println!(
                "{}s audio: CPU {:.2} ms, GPU {:.2} ms, speedup x{:.2}",
                seconds,
                cpu_elapsed.as_secs_f64() * 1000.0,
                gpu_elapsed.as_secs_f64() * 1000.0,
                speedup
            );
        }
    }
}
