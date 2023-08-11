use crate::mel::interleave_frames;
use crate::quant::to_array2;
use crate::quant::{dequantize, quantize};
use fast_image_resize as fr;
use ndarray::Array2;
use std::num::NonZeroU32;

pub fn downscale_frames(frames: &[Array2<f64>], factor: usize) -> Array2<f64> {
    let (requant, range) = quantize(&interleave_frames(&frames, false, 0));
    let shape = frames[0].raw_dim();
    let n_mels = shape[0];
    let width = NonZeroU32::new((requant.len() / n_mels) as u32).unwrap();
    let height = NonZeroU32::new(n_mels as u32).unwrap();
    let src_image =
        fr::Image::from_vec_u8(width, height, requant.clone(), fr::PixelType::U8).unwrap();
    let dst_width = NonZeroU32::new(((requant.len() / n_mels) / factor) as u32).unwrap();
    let dst_height = NonZeroU32::new((n_mels / factor) as u32).unwrap();
    let mut dst_image = fr::Image::new(dst_width, dst_height, src_image.pixel_type());
    let mut dst_view = dst_image.view_mut();
    let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    resizer.resize(&src_image.view(), &mut dst_view).unwrap();
    let dequantized_resized = dequantize(dst_image.buffer(), &range);
    let resized_frames = to_array2(&dequantized_resized, n_mels / factor);
    resized_frames
}
