use std::thread::spawn;

use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array2;

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

pub fn decompose(
    data: &mut Array2<f32>,
    pixel_scale: usize,
    width: usize,
    height: usize,
) -> Array2<f32> {
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
                    let kernel_value = kernel_values[(kernel_index_x + abs_kernel_size) as usize]
                        [(kernel_index_y + abs_kernel_size) as usize];

                    pixels_sum += kernel_value * data[index];
                }
            }

            current_data[[y, x]] = pixels_sum;
        }
    }

    let final_data = data.clone() - &current_data;
    *data = current_data;

    final_data
}

fn get_min_value(data: &Array2<f32>) -> f32 {
    *data
        .iter()
        .reduce(|current, previous| {
            if current < previous {
                current
            } else {
                previous
            }
        })
        .unwrap()
}

fn get_max_value(data: &Array2<f32>) -> f32 {
    *data
        .iter()
        .reduce(|current, previous| {
            if current > previous {
                current
            } else {
                previous
            }
        })
        .unwrap()
}

pub fn a_trous_transform(image: &DynamicImage, levels: usize) {
    let mut pixel_scale = 0;

    let image = image.to_rgb32f();
    let (width, height) = (image.width() as usize, image.height() as usize);

    let mut data_r = Array2::<f32>::zeros((height, width));
    let mut data_g = Array2::<f32>::zeros((height, width));
    let mut data_b = Array2::<f32>::zeros((height, width));

    for (x, y, pixel) in image.enumerate_pixels() {
        let [r, g, b] = pixel.0;

        data_r[[y as usize, x as usize]] = r;
        data_g[[y as usize, x as usize]] = g;
        data_b[[y as usize, x as usize]] = b;
    }

    while pixel_scale < levels {
        let handler_r = spawn(move || {
            let final_r = decompose(&mut data_r, pixel_scale, width, height);
            (data_r, final_r)
        });

        let handler_g = spawn(move || {
            let final_g = decompose(&mut data_g, pixel_scale, width, height);
            (data_g, final_g)
        });

        let handler_b = spawn(move || {
            let final_b = decompose(&mut data_b, pixel_scale, width, height);
            (data_b, final_b)
        });

        let (data_r_copy, final_r) = handler_r.join().unwrap();
        data_r = data_r_copy;
        let (data_g_copy, final_g) = handler_g.join().unwrap();
        data_g = data_g_copy;
        let (data_b_copy, final_b) = handler_b.join().unwrap();
        data_b = data_b_copy;

        let min_r = get_min_value(&final_r);
        let min_g = get_min_value(&final_g);
        let min_b = get_min_value(&final_b);

        let min_pixel = min_r.min(min_g).min(min_b);

        let max_r = get_max_value(&final_r);
        let max_g = get_max_value(&final_g);
        let max_b = get_max_value(&final_b);

        let max_pixel = max_r.max(max_g).max(max_b);

        let mut result_img: ImageBuffer<Rgb<f32>, Vec<f32>> =
            ImageBuffer::new(width as u32, height as u32);

        let rescale_ratio = max_pixel - min_pixel;

        for (x, y, pixel) in result_img.enumerate_pixels_mut() {
            let red = final_r[(y as usize, x as usize)];
            let green = final_g[(y as usize, x as usize)];
            let blue = final_b[(y as usize, x as usize)];

            let scaled_red = (red - min_pixel) / rescale_ratio;
            let scaled_green = (green - min_pixel) / rescale_ratio;
            let scaled_blue = (blue - min_pixel) / rescale_ratio;

            *pixel = Rgb([scaled_red, scaled_green, scaled_blue]);
        }

        DynamicImage::ImageRgb32F(result_img)
            .to_rgb8()
            .save(format!("level{pixel_scale}.jpg"))
            .unwrap();

        pixel_scale += 1;
    }

    let min_r = get_min_value(&data_r);
    let min_g = get_min_value(&data_g);
    let min_b = get_min_value(&data_b);

    let min_pixel = min_r.min(min_g).min(min_b);

    let max_r = get_max_value(&data_r);
    let max_g = get_max_value(&data_g);
    let max_b = get_max_value(&data_b);

    let max_pixel = max_r.max(max_g).max(max_b);

    let mut result_img: ImageBuffer<Rgb<f32>, Vec<f32>> =
        ImageBuffer::new(width as u32, height as u32);

    let rescale_ratio = max_pixel - min_pixel;

    for (x, y, pixel) in result_img.enumerate_pixels_mut() {
        let red = data_r[(y as usize, x as usize)];
        let green = data_g[(y as usize, x as usize)];
        let blue = data_b[(y as usize, x as usize)];

        let scaled_red = (red - min_pixel) / rescale_ratio;
        let scaled_green = (green - min_pixel) / rescale_ratio;
        let scaled_blue = (blue - min_pixel) / rescale_ratio;

        *pixel = Rgb([scaled_red, scaled_green, scaled_blue]);
    }

    DynamicImage::ImageRgb32F(result_img)
        .to_rgb8()
        .save("residue.jpg")
        .unwrap();
}
