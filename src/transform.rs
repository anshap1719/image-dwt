use image::DynamicImage;
use ndarray::{Array2, Array3};

use crate::aggregate::Aggregate;
use crate::decompose::WaveletDecompose;
use crate::kernels::{B3SplineKernel, Kernel, LinearInterpolationKernel, LowScaleKernel};
use crate::layer::{WaveletLayer, WaveletLayerBuffer};

#[derive(Copy, Clone)]
pub struct Scale {
    min: f32,
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

pub struct ATrousTransform<const KERNEL_SIZE: usize, KernelType: Kernel<KERNEL_SIZE> + 'static> {
    input: ATrousTransformInput,
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

        match &mut self.input {
            ATrousTransformInput::Grayscale { data } => {
                if pixel_scale == self.levels {
                    return Some(WaveletLayer {
                        buffer: WaveletLayerBuffer::Grayscale { data: data.clone() },
                        pixel_scale: None,
                    });
                }

                let (width, height) = (self.width, self.height);
                let kernel = self.kernel;

                let layer_buffer = data.wavelet_decompose(kernel, pixel_scale, width, height);
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

                let (width, height) = (self.width, self.height);
                let kernel = self.kernel;

                let layer_buffer = data.wavelet_decompose(kernel, pixel_scale, width, height);
                Some(WaveletLayer {
                    pixel_scale: Some(pixel_scale),
                    buffer: layer_buffer,
                })
            }
        }
    }
}
