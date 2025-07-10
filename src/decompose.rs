use crate::kernels::{B3SplineKernel, Kernel, LinearInterpolationKernel};
use convolve_image::Convolution;
use ndarray::{Array2, Array3};

use crate::layer::WaveletLayerBuffer;

pub trait WaveletDecompose {
    fn wavelet_decompose(
        &mut self,
        kernel: Kernel,
        pixel_scale: usize,
        is_raw: bool,
    ) -> WaveletLayerBuffer;
}

impl WaveletDecompose for Array2<f32> {
    fn wavelet_decompose(
        &mut self,
        kernel: Kernel,
        pixel_scale: usize,
        _: bool,
    ) -> WaveletLayerBuffer {
        let stride = 2_usize.pow(
            u32::try_from(pixel_scale)
                .unwrap_or_else(|_| panic!("pixel_scale cannot be larger than {}", u32::MAX)),
        );
        let mut current_data = self.clone();
        match kernel {
            Kernel::LinearInterpolationKernel => {
                current_data.convolve(LinearInterpolationKernel::new().into(), stride);
            }
            Kernel::LowScaleKernel => {
                unimplemented!("Low scale is not a separable kernel");
            }
            Kernel::B3SplineKernel => current_data.convolve(B3SplineKernel::new().into(), stride),
        }

        let final_data = self.clone() - &current_data;
        *self = current_data;

        WaveletLayerBuffer::Grayscale { data: final_data }
    }
}

impl WaveletDecompose for Array3<f32> {
    fn wavelet_decompose(
        &mut self,
        kernel: Kernel,
        pixel_scale: usize,
        is_raw: bool,
    ) -> WaveletLayerBuffer {
        let stride = 2_usize.pow(
            u32::try_from(pixel_scale)
                .unwrap_or_else(|_| panic!("pixel_scale cannot be larger than {}", u32::MAX)),
        );
        let mut current_data = self.clone();
        match kernel {
            Kernel::LinearInterpolationKernel => {
                current_data.convolve(LinearInterpolationKernel::new().into(), stride);
            }
            Kernel::LowScaleKernel => {
                unimplemented!("Low scale is not a separable kernel");
            }
            Kernel::B3SplineKernel => current_data.convolve(B3SplineKernel::new().into(), stride),
        }

        let final_data = self.clone() - &current_data;
        *self = current_data;

        if is_raw {
            WaveletLayerBuffer::Raw { data: final_data }
        } else {
            WaveletLayerBuffer::Rgb { data: final_data }
        }
    }
}
