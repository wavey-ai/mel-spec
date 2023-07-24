use std::fs::File;
use std::io::{self, Read, Write};

#[derive(Debug)]
pub struct QuantizationRange {
    pub min: f32,
    pub max: f32,
}

pub fn save_tga_8bit(data: &[f32], n_mels: usize, path: &str) -> io::Result<()> {
    let width = (data.len() / n_mels) as u16;
    let height = n_mels as u16;
    let tga_size = (width as usize) * (height as usize);

    // Quantize the floating-point data to 8-bit grayscale
    let (tga_data, range) = quantize(&data.to_vec());

    assert!(tga_size == tga_data.len(), "tga data wrong size");
    // Convert the quantized data to 8-bit color index (0-255) and store in tga_data

    // TGA Header (18 bytes)
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

    // Combine the TGA header and image data and save to the file
    let mut tga_image = Vec::new();
    tga_image.extend_from_slice(&tga_header);
    tga_image.extend_from_slice(&tga_data);

    let mut file = File::create(path)?;
    file.write_all(&tga_image)?;

    Ok(())
}

pub fn load_tga_8bit(path: &str) -> io::Result<Vec<f32>> {
    let mut file = File::open(path)?;
    let mut tga_data = Vec::new();
    let mut min_bytes = [0u8; 4];
    let mut max_bytes = [0u8; 4];

    // discard TGA header
    file.read_exact(&mut [0; 18])?;

    file.read_exact(&mut min_bytes)?;
    file.read_exact(&mut max_bytes)?;

    let min = f32::from_le_bytes(min_bytes);
    let max = f32::from_le_bytes(max_bytes);

    let range = QuantizationRange { min, max };

    // Read the rest of the file (image data) into the tga_data vector
    file.read_to_end(&mut tga_data)?;

    let mel = dequantize(&tga_data, &range);

    Ok(mel)
}

pub fn quantize(frame: &Vec<f32>) -> (Vec<u8>, QuantizationRange) {
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

pub fn dequantize(data: &[u8], range: &QuantizationRange) -> Vec<f32> {
    let mut result = Vec::with_capacity(data.len());
    let scale = (range.max - range.min) / 255.0;

    for &value in data {
        let scaled_value = value as f32 * scale + range.min;
        result.push(scaled_value);
    }

    result
}
