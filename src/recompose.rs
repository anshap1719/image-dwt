use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array2;

use crate::aggregate::Aggregate;
use crate::layer::WaveletLayer;
use crate::transform::ChannelWiseData;

pub trait RecomposableWaveletLayers: Iterator<Item = WaveletLayer> {
    fn recompose_into_image(self, width: usize, height: usize) -> DynamicImage
    where
        Self: Sized,
    {
        let mut result = ChannelWiseData {
            red: Array2::<f32>::zeros((height, width)),
            green: Array2::<f32>::zeros((height, width)),
            blue: Array2::<f32>::zeros((height, width)),
        };

        for layer in self {
            for (x, y, pixel) in layer.image_buffer.enumerate_pixels() {
                let x = x as usize;
                let y = y as usize;

                let [red, green, blue] = pixel.0;
                result.red[[y, x]] += red;
                result.green[[y, x]] += green;
                result.blue[[y, x]] += blue;
            }
        }

        let min_r = result.red.min();
        let min_g = result.green.min();
        let min_b = result.blue.min();

        let min_pixel = min_r.min(min_g).min(min_b);

        let max_r = result.red.max();
        let max_g = result.green.max();
        let max_b = result.blue.max();

        let max_pixel = max_r.max(max_g).max(max_b);

        let mut result_img: ImageBuffer<Rgb<f32>, Vec<f32>> =
            ImageBuffer::new(width as u32, height as u32);

        let rescale_ratio = max_pixel - min_pixel;

        for (x, y, pixel) in result_img.enumerate_pixels_mut() {
            let red = result.red[(y as usize, x as usize)];
            let green = result.green[(y as usize, x as usize)];
            let blue = result.blue[(y as usize, x as usize)];

            let scaled_red = (red - min_pixel) / rescale_ratio;
            let scaled_green = (green - min_pixel) / rescale_ratio;
            let scaled_blue = (blue - min_pixel) / rescale_ratio;

            *pixel = Rgb([scaled_red, scaled_green, scaled_blue]);
        }

        DynamicImage::ImageRgb32F(result_img)
    }
}

impl<T> RecomposableWaveletLayers for T where T: Iterator<Item = WaveletLayer> {}
