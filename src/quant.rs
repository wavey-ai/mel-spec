use ndarray::{s, Array2};
use std::fs::File;
use std::io::{self, Read, Write};

#[derive(Debug)]
pub struct QuantizationRange {
    pub min: f32,
    pub max: f32,
}

/// Create a TARGA format image data for the spectrogram. `data` must be interleaved in major row order:
/// use ['mel::interleave_frames`] on the spectrogram first.
///
/// The min/max range used in quantization is stored in the header.
pub fn save_tga_8bit(data: &[f32], n_mels: usize, path: &str) -> io::Result<()> {
    let width = (data.len() / n_mels) as u16;
    assert!(
        width < u16::MAX,
        "width greater than TARGA max, use [`tga_8bit`]"
    );

    let data = tga_8bit_data(&data, n_mels);
    let mut file = File::create(path)?;
    file.write_all(&data)?;

    Ok(())
}

pub fn tga_8bit(data: &[f32], n_mels: usize) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    for chunk in chunk_frames_into_strides(data.to_vec(), n_mels, u16::MAX as usize) {
        result.push(tga_8bit_data(&chunk, n_mels));
    }

    result
}

pub fn tga_8bit_data(data: &[f32], n_mels: usize) -> Vec<u8> {
    // Quantize the floating-point data to 8-bit grayscale
    let (tga_data, range) = quantize(&data.to_vec());

    let width = (data.len() / n_mels) as u16;
    let height = n_mels as u16;

    // Combine the TGA header and image data
    let mut tga_header = Vec::with_capacity(18);
    tga_header.push(8u8); // ID len
    tga_header.push(0u8); // color map type (unused)
    tga_header.push(3u8); // Uncompressed, black and white images.
    tga_header.extend_from_slice(&[0u8; 5]); // color map spec (unused)
    tga_header.extend_from_slice(&[0u8; 4]); // X and Y Origin (unused)
    tga_header.extend_from_slice(&width.to_le_bytes()); // Image Width (little-endian)
    tga_header.extend_from_slice(&height.to_le_bytes()); // Image Height (little-endian)
    tga_header.push(8u8); // Bits per Pixel (8 bits)
    tga_header.push(0u8);
    tga_header.extend_from_slice(&range.min.to_le_bytes());
    tga_header.extend_from_slice(&range.max.to_le_bytes());

    let mut tga_image = Vec::new();
    tga_image.extend_from_slice(&tga_header);
    tga_image.extend_from_slice(&tga_data);

    tga_image
}

pub fn parse_tga_8bit(data: &[u8]) -> io::Result<Vec<f32>> {
    let mut cursor = io::Cursor::new(data);
    let mut tga_data = Vec::new();
    let mut min_bytes = [0u8; 4];
    let mut max_bytes = [0u8; 4];

    // Discard TGA header (18 bytes)
    cursor.set_position(18);

    cursor.read_exact(&mut min_bytes)?;
    cursor.read_exact(&mut max_bytes)?;

    let min = f32::from_le_bytes(min_bytes);
    let max = f32::from_le_bytes(max_bytes);

    let range = QuantizationRange { min, max };

    cursor.read_to_end(&mut tga_data)?;

    let mel = dequantize(&tga_data, &range);

    Ok(mel)
}

/// Load a TARGA file from disk, returning the interleaved frame data.
pub fn load_tga_8bit(path: &str) -> io::Result<Vec<f32>> {
    let mut file = File::open(path)?;
    let mut tga_data = Vec::new();
    file.read_to_end(&mut tga_data)?;

    parse_tga_8bit(&tga_data)
}

/// Utility function to chunk a major row-order interleaved spectrogram
pub fn chunk_frames_into_strides(
    frames: Vec<f32>,
    n_mels: usize,
    stride_size: usize,
) -> Vec<Vec<f32>> {
    let width = frames.len() / n_mels;

    if stride_size == width {
        return vec![frames];
    }

    let height = n_mels;

    // Create a 2D ndarray from the image data
    let ndarray_image = Array2::from_shape_vec((height, width), frames).unwrap();

    // Create a vector to store the chunks
    let mut chunks = Vec::new();

    // Chunk the frames array into smaller strides
    for y in (0..height).step_by(stride_size) {
        for x in (0..width).step_by(stride_size) {
            let end_y = (y + stride_size).min(height);
            let end_x = (x + stride_size).min(width);

            // Create a 2D slice representing the chunk and flatten it into a Vec<f32>
            let chunk = ndarray_image
                .slice(s![y..end_y, x..end_x])
                .to_owned()
                .into_raw_vec_and_offset()
                .0;
            chunks.push(chunk);
        }
    }

    chunks
}

/// Quantize an interleaved spectrogram, returning u8 bytes suitable for
/// grayscale.
pub fn quantize(frame: &[f32]) -> (Vec<u8>, QuantizationRange) {
    let mut result: Vec<u8> = Vec::new();
    let min = frame.iter().copied().fold(f32::INFINITY, f32::min);
    let max = frame.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = 255.0 / (max - min);

    for &value in frame {
        let scaled_value = ((value - min) * scale).round().max(0.0).min(255.0);
        result.push(scaled_value as u8);
    }

    (result, QuantizationRange { min, max })
}

/// Dequantize u8 bytes to original f32 values.
pub fn dequantize(data: &[u8], range: &QuantizationRange) -> Vec<f32> {
    let mut result = Vec::with_capacity(data.len());
    let scale = (range.max - range.min) / 255.0;

    for &value in data {
        let scaled_value = value as f32 * scale + range.min;
        result.push(scaled_value);
    }

    result
}

/// De-interleave from major row order back into an `Array2<f64>`
pub fn to_array2(frames: &[f32], n_mels: usize) -> Array2<f64> {
    Array2::from_shape_vec(
        (n_mels, frames.len() / n_mels),
        frames.iter().map(|v| *v as f64).collect::<Vec<_>>(),
    )
    .unwrap()
}
