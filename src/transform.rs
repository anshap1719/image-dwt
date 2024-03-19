use std::thread::spawn;

use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array2;

use crate::extended_index::ExtendedIndex;
use crate::kernels::{B3SplineKernel, Kernel, LinearInterpolationKernel, LowScaleKernel};
use crate::layer::{WaveletLayer, WaveletLayerBuffer};

pub struct TransformData {
    red: Array2<f32>,
    green: Array2<f32>,
    blue: Array2<f32>,
}

pub struct ATrousTransform<const KERNEL_SIZE: usize, KernelType: Kernel<KERNEL_SIZE> + 'static> {
    input: TransformData,
    levels: usize,
    kernel: KernelType,
    current_level: usize,
    width: usize,
    height: usize,
}

impl<const KERNEL_SIZE: usize, KernelType: Kernel<KERNEL_SIZE>>
    ATrousTransform<KERNEL_SIZE, KernelType>
{
    pub fn new(input: &DynamicImage, levels: usize, kernel: KernelType) -> Self {
        let (width, height) = (input.width() as usize, input.height() as usize);

        let mut data_r = Array2::<f32>::zeros((height, width));
        let mut data_g = Array2::<f32>::zeros((height, width));
        let mut data_b = Array2::<f32>::zeros((height, width));

        let input = input.to_rgb32f();

        for (x, y, pixel) in input.enumerate_pixels() {
            let [r, g, b] = pixel.0;

            data_r[[y as usize, x as usize]] = r;
            data_g[[y as usize, x as usize]] = g;
            data_b[[y as usize, x as usize]] = b;
        }

        let input = TransformData {
            red: data_r,
            green: data_g,
            blue: data_b,
        };

        Self {
            input,
            width,
            height,
            levels,
            kernel,
            current_level: 0,
        }
    }

    pub fn linear(
        input: &DynamicImage,
        levels: usize,
    ) -> ATrousTransform<3, LinearInterpolationKernel> {
        ATrousTransform::<3, LinearInterpolationKernel>::new(
            input,
            levels,
            LinearInterpolationKernel,
        )
    }

    pub fn low_scale(input: &DynamicImage, levels: usize) -> ATrousTransform<3, LowScaleKernel> {
        ATrousTransform::<3, LowScaleKernel>::new(input, levels, LowScaleKernel)
    }

    pub fn b_spline(input: &DynamicImage, levels: usize) -> ATrousTransform<5, B3SplineKernel> {
        ATrousTransform::<5, B3SplineKernel>::new(input, levels, B3SplineKernel)
    }
}

impl<const KERNEL_SIZE: usize, KernelType: Kernel<KERNEL_SIZE>> Iterator
    for ATrousTransform<KERNEL_SIZE, KernelType>
{
    type Item = WaveletLayer;

    fn next(&mut self) -> Option<Self::Item> {
        let pixel_scale = self.current_level;
        self.current_level += 1;

        if pixel_scale > self.levels {
            return None;
        }

        if pixel_scale == self.levels {
            let min_r = get_min_value(&self.input.red);
            let min_g = get_min_value(&self.input.green);
            let min_b = get_min_value(&self.input.blue);

            let min_pixel = min_r.min(min_g).min(min_b);

            let max_r = get_max_value(&self.input.red);
            let max_g = get_max_value(&self.input.green);
            let max_b = get_max_value(&self.input.blue);

            let max_pixel = max_r.max(max_g).max(max_b);

            let mut result_img: ImageBuffer<Rgb<f32>, Vec<f32>> =
                ImageBuffer::new(self.width as u32, self.height as u32);

            let rescale_ratio = max_pixel - min_pixel;

            for (x, y, pixel) in result_img.enumerate_pixels_mut() {
                let red = self.input.red[(y as usize, x as usize)];
                let green = self.input.green[(y as usize, x as usize)];
                let blue = self.input.blue[(y as usize, x as usize)];

                let scaled_red = (red - min_pixel) / rescale_ratio;
                let scaled_green = (green - min_pixel) / rescale_ratio;
                let scaled_blue = (blue - min_pixel) / rescale_ratio;

                *pixel = Rgb([scaled_red, scaled_green, scaled_blue]);
            }

            return Some(WaveletLayer {
                image_buffer: result_img,
                pixel_scale: None,
            });
        }

        let (width, height) = (self.width, self.height);

        let mut data_r = self.input.red.clone();
        let mut data_g = self.input.green.clone();
        let mut data_b = self.input.blue.clone();

        let kernel = self.kernel;

        let handler_r = spawn(move || {
            let final_r = decompose(&mut data_r, kernel, pixel_scale, width, height);
            (data_r, final_r)
        });

        let handler_g = spawn(move || {
            let final_g = decompose(&mut data_g, kernel, pixel_scale, width, height);
            (data_g, final_g)
        });

        let handler_b = spawn(move || {
            let final_b = decompose(&mut data_b, kernel, pixel_scale, width, height);
            (data_b, final_b)
        });

        let (data_r_copy, final_r) = handler_r.join().unwrap();
        self.input.red = data_r_copy;
        let (data_g_copy, final_g) = handler_g.join().unwrap();
        self.input.green = data_g_copy;
        let (data_b_copy, final_b) = handler_b.join().unwrap();
        self.input.blue = data_b_copy;

        let min_r = get_min_value(&final_r.data);
        let min_g = get_min_value(&final_g.data);
        let min_b = get_min_value(&final_b.data);

        let min_pixel = min_r.min(min_g).min(min_b);

        let max_r = get_max_value(&final_r.data);
        let max_g = get_max_value(&final_g.data);
        let max_b = get_max_value(&final_b.data);

        let max_pixel = max_r.max(max_g).max(max_b);

        let mut result_img: ImageBuffer<Rgb<f32>, Vec<f32>> =
            ImageBuffer::new(width as u32, height as u32);

        let rescale_ratio = max_pixel - min_pixel;

        for (x, y, pixel) in result_img.enumerate_pixels_mut() {
            let red = final_r.data[(y as usize, x as usize)];
            let green = final_g.data[(y as usize, x as usize)];
            let blue = final_b.data[(y as usize, x as usize)];

            let scaled_red = (red - min_pixel) / rescale_ratio;
            let scaled_green = (green - min_pixel) / rescale_ratio;
            let scaled_blue = (blue - min_pixel) / rescale_ratio;

            *pixel = Rgb([scaled_red, scaled_green, scaled_blue]);
        }

        Some(WaveletLayer {
            pixel_scale: Some(pixel_scale),
            image_buffer: result_img,
        })
    }
}

pub fn decompose<const KERNEL_SIZE: usize>(
    data: &mut Array2<f32>,
    kernel: impl Kernel<KERNEL_SIZE>,
    pixel_scale: usize,
    width: usize,
    height: usize,
) -> WaveletLayerBuffer {
    let distance = 2_usize.pow(pixel_scale as u32);
    let mut current_data = Array2::<f32>::zeros((height, width));

    for x in 0..width {
        for y in 0..height {
            let mut pixels_sum = 0.0;

            let abs_kernel_size = (kernel.size() / 2) as isize;
            let kernel_values = kernel.values();

            for kernel_index_x in -abs_kernel_size..=abs_kernel_size {
                for kernel_index_y in -abs_kernel_size..=abs_kernel_size {
                    let index = current_data.compute_extended_index(
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

    WaveletLayerBuffer {
        data: final_data,
        pixel_scale,
    }
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
