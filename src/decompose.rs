use ndarray::Array2;

use crate::kernels::Kernel;
use crate::layer::WaveletLayerBuffer;

pub trait WaveletDecompose {
    fn wavelet_decompose<const KERNEL_SIZE: usize>(
        &mut self,
        kernel: impl Kernel<KERNEL_SIZE>,
        pixel_scale: usize,
        width: usize,
        height: usize,
    ) -> WaveletLayerBuffer;
}

impl WaveletDecompose for Array2<f32> {
    fn wavelet_decompose<const KERNEL_SIZE: usize>(
        &mut self,
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
                        let index = kernel.compute_extended_index(
                            x,
                            y,
                            kernel_index_x * distance as isize,
                            kernel_index_y * distance as isize,
                            &current_data,
                        );
                        let kernel_value = kernel_values
                            [(kernel_index_x + abs_kernel_size) as usize]
                            [(kernel_index_y + abs_kernel_size) as usize];

                        pixels_sum += kernel_value * self[index];
                    }
                }

                current_data[[y, x]] = pixels_sum;
            }
        }

        let final_data = self.clone() - &current_data;
        *self = current_data;

        WaveletLayerBuffer { data: final_data }
    }
}
