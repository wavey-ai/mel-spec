use ndarray::{s, Array, Array2, Zip};

#[derive(Debug)]
pub struct EdgeInfo {
    pub gradient_count: u32,
    pub non_intersected_columns: Vec<usize>,
}

/// turns out edge detection is a reliable way of doing word boundary detection in real-time
fn edge_detect(
    frames: &[f32],
    n_mels: usize,
    threshold: f32,
    min_intersections: usize,
) -> EdgeInfo {
    let width = frames.len() / n_mels;
    let height = n_mels;

    // Create a 2D ndarray from the image data
    let ndarray_image = Array2::from_shape_vec((height, width), frames.to_vec()).unwrap();

    // Sobel kernels for gradient calculation
    let sobel_x =
        Array::from_shape_vec((3, 3), vec![-1., 0., 1., -2., 0., 2., -1., 0., 1.]).unwrap();
    let sobel_y =
        Array::from_shape_vec((3, 3), vec![-1., -2., -1., 0., 0., 0., 1., 2., 1.]).unwrap();

    // Convolve with Sobel kernels and calculate the gradient magnitude
    let gradient_mag = Array::from_shape_fn((height - 2, width - 2), |(y, x)| {
        let view = ndarray_image.slice(s![y..y + 3, x..x + 3]);
        let gradient_x = Zip::from(view)
            .and(&sobel_x)
            .fold(0.0, |acc, &a, &b| acc + a * b);
        let gradient_y = Zip::from(view)
            .and(&sobel_y)
            .fold(0.0, |acc, &a, &b| acc + a * b);
        (gradient_x * gradient_x + gradient_y * gradient_y).sqrt()
    });

    // Count the number of high-energy gradients above the threshold
    let gradient_count = gradient_mag.iter().filter(|&val| *val >= threshold).count() as u32;

    // Identify columns with no gradient intersection
    let mut non_intersected_columns: Vec<usize> = Vec::new();
    for x in 0..width - 2 {
        let num_intersections = (0..height - 2)
            .filter(|&y| gradient_mag[(y, x)] >= threshold)
            .count();
        if num_intersections < min_intersections {
            non_intersected_columns.push(x);
        }
    }

    EdgeInfo {
        gradient_count,
        non_intersected_columns,
    }
}

mod tests {
    use super::*;
    use crate::mel::chunk_frames;
    use crate::quant::load_tga_8bit;
    use image::{ImageBuffer, Rgb};
    use std::path::Path;
    use std::time::Instant;

    fn as_image(
        image: &Array2<f32>,
        non_intersected_columns: &[usize],
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (height, width) = image.dim();
        let mut img_buffer = ImageBuffer::new(width as u32, height as u32);

        // Scale the values to the range [0, 255] for the grayscale image
        let max_val = image.fold(0.0, |acc: f32, &val| acc.max(val));
        let scaled_image: Array2<u8> = image.mapv(|val| (val * (255.0 / max_val)) as u8);

        for (y, row) in scaled_image.outer_iter().rev().enumerate() {
            for (x, &val) in row.into_iter().enumerate() {
                // Create an RGB pixel with all color channels set to the grayscale value
                let rgb_pixel = Rgb([val, val, val]);
                // Apply the green tint to the color channel for interesting columns
                if non_intersected_columns.contains(&x) {
                    // Check if the current row is within the top 10 rows of the image
                    if y < 10 {
                        // Set the pixel to be entirely green for the top 10 rows
                        let green_tint = Rgb([0, 255, 0]);
                        img_buffer.put_pixel(x as u32, y as u32, green_tint);
                    } else {
                        // Apply a subtle green tint to the pixel for the rest of the rows
                        let green_tint_value = 60;
                        let green_tint = Rgb([val, val + green_tint_value, val]);
                        img_buffer.put_pixel(x as u32, y as u32, green_tint);
                    }
                } else {
                    // No tint for non-interesting columns, keep the grayscale value
                    img_buffer.put_pixel(x as u32, y as u32, rgb_pixel);
                }
            }
        }

        img_buffer
    }

    #[test]
    fn test_extract_edges() {
        let n_mels = 80;
        let file_path = "./test/quantized_mel_golden.tga";
        let dequantized_mel = load_tga_8bit(file_path).unwrap();
        let start_time = Instant::now();
        let edge_info = edge_detect(&dequantized_mel, n_mels, 1.0, 5);
        let elapsed_time = start_time.elapsed().as_millis();
        dbg!(elapsed_time);
        let file_path = "./doc/jfk_vad_boundaries.png";
        let gradient_mag =
            Array2::from_shape_vec((n_mels, dequantized_mel.len() / n_mels), dequantized_mel)
                .unwrap();
        let non_intersected_columns = edge_info.non_intersected_columns;
        let img_buffer = as_image(&gradient_mag, &non_intersected_columns);
        img_buffer.save(file_path).unwrap();
    }
}
