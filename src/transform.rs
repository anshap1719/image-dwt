use image::DynamicImage;
use ndarray::{Array2, Array3};

use crate::aggregate::Aggregate;
use crate::decompose::WaveletDecompose;
use crate::kernels::Kernel;
use crate::layer::{WaveletLayer, WaveletLayerBuffer};

#[derive(Copy, Clone)]
pub struct Scale {
    min: f32,
    #[allow(unused)]
    max: f32,
    scaling_ratio: f32,
}

impl Scale {
    pub fn new(min: f32, max: f32) -> Self {
        Self {
            min,
            max,
            scaling_ratio: max - min,
        }
    }

    #[inline]
    pub fn apply(&self, value: f32) -> f32 {
        (value - self.min) / self.scaling_ratio
    }
}

#[derive(Clone)]
enum ATrousTransformInput {
    Grayscale { data: Array2<f32> },
    Rgb { data: Array3<f32> },
}

impl Aggregate for ATrousTransformInput {
    fn min(&self) -> f32 {
        match self {
            ATrousTransformInput::Grayscale { data } => data.min(),
            ATrousTransformInput::Rgb { data } => data.min(),
        }
    }

    fn max(&self) -> f32 {
        match self {
            ATrousTransformInput::Grayscale { data } => data.max(),
            ATrousTransformInput::Rgb { data } => data.max(),
        }
    }
}

#[derive(Clone)]
pub struct ATrousTransform {
    input: ATrousTransformInput,
    levels: usize,
    kernel: Kernel,
    current_level: usize,
}

impl ATrousTransform {
    pub fn new(input: &DynamicImage, levels: usize, kernel: Kernel) -> Self {
        let (width, height) = (input.width() as usize, input.height() as usize);

        let input = match &input {
            DynamicImage::ImageLuma8(_)
            | DynamicImage::ImageLumaA8(_)
            | DynamicImage::ImageLuma16(_)
            | DynamicImage::ImageLumaA16(_) => {
                let mut data = Array2::zeros((height, width));
                for (x, y, pixel) in input.to_luma32f().enumerate_pixels() {
                    data[[y as usize, x as usize]] = pixel.0[0];
                }

                ATrousTransformInput::Grayscale { data }
            }
            _ => {
                let mut data = Array3::zeros((height, width, 3));
                for (x, y, pixel) in input.to_rgb32f().enumerate_pixels() {
                    let [red, green, blue] = pixel.0;
                    data[[y as usize, x as usize, 0]] = red;
                    data[[y as usize, x as usize, 1]] = green;
                    data[[y as usize, x as usize, 2]] = blue;
                }

                ATrousTransformInput::Rgb { data }
            }
        };

        Self {
            input,
            levels,
            kernel,
            current_level: 0,
        }
    }

    pub fn linear(input: &DynamicImage, levels: usize) -> Self {
        ATrousTransform::new(input, levels, Kernel::LinearInterpolationKernel)
    }

    pub fn low_scale(input: &DynamicImage, levels: usize) -> Self {
        ATrousTransform::new(input, levels, Kernel::LowScaleKernel)
    }

    pub fn b_spline(input: &DynamicImage, levels: usize) -> Self {
        ATrousTransform::new(input, levels, Kernel::B3SplineKernel)
    }
}

impl Iterator for ATrousTransform {
    type Item = WaveletLayer;

    fn next(&mut self) -> Option<Self::Item> {
        let pixel_scale = self.current_level;
        self.current_level += 1;

        if pixel_scale > self.levels {
            return None;
        }

        match &mut self.input {
            ATrousTransformInput::Grayscale { data } => {
                if pixel_scale == self.levels {
                    return Some(WaveletLayer {
                        buffer: WaveletLayerBuffer::Grayscale { data: data.clone() },
                        pixel_scale: None,
                    });
                }

                let kernel = self.kernel;

                let layer_buffer = data.wavelet_decompose(kernel, pixel_scale);
                Some(WaveletLayer {
                    pixel_scale: Some(pixel_scale),
                    buffer: layer_buffer,
                })
            }
            ATrousTransformInput::Rgb { data } => {
                if pixel_scale == self.levels {
                    return Some(WaveletLayer {
                        buffer: WaveletLayerBuffer::Rgb { data: data.clone() },
                        pixel_scale: None,
                    });
                }

                let kernel = self.kernel;

                let layer_buffer = data.wavelet_decompose(kernel, pixel_scale);
                Some(WaveletLayer {
                    pixel_scale: Some(pixel_scale),
                    buffer: layer_buffer,
                })
            }
        }
    }
}
