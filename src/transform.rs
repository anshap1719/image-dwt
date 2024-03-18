use std::fs::File;

use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::Array2;
use tiff::encoder::{colortype, TiffEncoder};

use crate::kernels::{Kernel, LinearInterpolationKernel};

pub fn compute_mirrored_index(max: usize, index: usize) -> usize {
    let mut index = index;

    if index > max - 1 {
        let diff = max.abs_diff(index);
        index = max - diff - 1;
    }

    index
}

trait VirtuallyMirrored {
    fn compute_index(&self, x: usize, y: usize, x_distance: isize, y_distance: isize)
        -> [usize; 2];
}

impl VirtuallyMirrored for Array2<f32> {
    fn compute_index(
        &self,
        x: usize,
        y: usize,
        x_distance: isize,
        y_distance: isize,
    ) -> [usize; 2] {
        let (max_y, max_x) = self.dim();

        let mut x = x as isize + x_distance;
        let mut y = y as isize + y_distance;

        if x < 0 {
            x = -x;
        } else if x > max_x as isize - 1 {
            let overshot_distance = x - max_x as isize + 1;
            x = max_x as isize - overshot_distance;
        }

        if y < 0 {
            y = -y;
        } else if y > max_y as isize - 1 {
            let overshot_distance = y - max_y as isize + 1;
            y = max_y as isize - overshot_distance;
        }

        [y as usize, x as usize]
    }
}

pub fn a_trous_transform(image: &DynamicImage, levels: usize) {
    let mut pixel_scale = 0;

    let image = image.to_luma32f();
    let (width, height) = (image.width() as usize, image.height() as usize);

    let mut data = Array2::<f32>::zeros((height, width));

    for (x, y, luminance) in image.enumerate_pixels() {
        data[[y as usize, x as usize]] = luminance.0[0]
    }

    let mut data = data;

    while pixel_scale < levels {
        let distance = 2_usize.pow(pixel_scale as u32);
        let mut current_data = Array2::<f32>::zeros((height, width));

        for x in 0..width {
            for y in 0..height {
                let mut pixels_sum = 0.0;

                let kernel = LinearInterpolationKernel;
                let abs_kernel_size = (kernel.size() / 2) as isize;
                let kernel_values = kernel.values();

                for kernel_index_x in -abs_kernel_size..=abs_kernel_size {
                    for kernel_index_y in -abs_kernel_size..=abs_kernel_size {
                        let index = current_data.compute_index(
                            x,
                            y,
                            kernel_index_x * distance as isize,
                            kernel_index_y * distance as isize,
                        );
                        let kernel_value = kernel_values
                            [(kernel_index_x + abs_kernel_size) as usize]
                            [(kernel_index_y + abs_kernel_size) as usize];

                        pixels_sum += kernel_value * data[index];
                    }
                }

                current_data[[y, x]] = pixels_sum;
            }
        }

        let final_data = &data - &current_data;

        let min_pixel = final_data
            .iter()
            .reduce(|current, previous| {
                if current < previous {
                    current
                } else {
                    previous
                }
            })
            .unwrap();

        let max_pixel = final_data
            .iter()
            .reduce(|current, previous| {
                if current > previous {
                    current
                } else {
                    previous
                }
            })
            .unwrap();

        let mut result_img: ImageBuffer<Luma<f32>, Vec<f32>> =
            ImageBuffer::new(width as u32, height as u32);

        println!("{min_pixel}, {max_pixel}");

        let rescale_ratio = max_pixel - min_pixel;

        println!("{rescale_ratio}");

        for (x, y, pixel) in result_img.enumerate_pixels_mut() {
            let val = final_data[(y as usize, x as usize)];
            *pixel = Luma([(val - min_pixel) / rescale_ratio]);
        }

        write_tiff(result_img, &format!("level{pixel_scale}"));

        pixel_scale += 1;

        data = current_data;
    }
}

pub fn write_tiff(input_image: ImageBuffer<Luma<f32>, Vec<f32>>, name: &str) {
    let output_file = File::create(format!("{name}.tif")).unwrap();
    let mut encoder = TiffEncoder::new(&output_file).unwrap();

    let mut image = encoder
        .new_image::<colortype::Gray32Float>(input_image.width(), input_image.height())
        .unwrap();

    let buffer = input_image.as_raw();

    image.rows_per_strip(4).unwrap();

    let mut idx = 0;
    while image.next_strip_sample_count() > 0 {
        let sample_count = image.next_strip_sample_count() as usize;
        image.write_strip(&buffer[idx..idx + sample_count]).unwrap();

        idx += sample_count;
    }

    image.finish().unwrap();
}
