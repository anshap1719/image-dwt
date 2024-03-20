use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array2;

use crate::aggregate::Aggregate;
use crate::transform::ChannelWiseData;

pub struct ATrousRecompose<'residue> {
    residue: &'residue DynamicImage,
    residue_bias: f32,
    width: usize,
    height: usize,
}

impl<'residue> ATrousRecompose<'residue> {
    pub fn new(residue: &'residue DynamicImage) -> Self {
        let (width, height) = (residue.width() as usize, residue.height() as usize);

        Self {
            residue,
            residue_bias: 1.,
            width,
            height,
        }
    }

    pub fn with_residue_bias(residue: &'residue DynamicImage, residue_bias: f32) -> Self {
        let (width, height) = (residue.width() as usize, residue.height() as usize);

        Self {
            residue,
            residue_bias,
            width,
            height,
        }
    }

    pub fn recompose(&self, layers: impl Iterator<Item = DynamicImage>) -> DynamicImage {
        let mut result = ChannelWiseData {
            red: Array2::<f32>::zeros((self.height, self.width)),
            green: Array2::<f32>::zeros((self.height, self.width)),
            blue: Array2::<f32>::zeros((self.height, self.width)),
        };

        for layer in layers {
            for (x, y, pixel) in layer.into_rgb32f().enumerate_pixels() {
                let x = x as usize;
                let y = y as usize;

                let [red, green, blue] = pixel.0;
                result.red[[y, x]] += red;
                result.green[[y, x]] += green;
                result.blue[[y, x]] += blue;
            }
        }

        for (x, y, pixel) in self.residue.to_rgb32f().enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;

            let [red, green, blue] = pixel.0;
            result.red[[y, x]] += red * self.residue_bias;
            result.green[[y, x]] += green * self.residue_bias;
            result.blue[[y, x]] += blue * self.residue_bias;
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
            ImageBuffer::new(self.width as u32, self.height as u32);

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

    pub fn recompose_biased(
        &self,
        layers: impl Iterator<Item = (DynamicImage, f32)>,
    ) -> DynamicImage {
        let mut result = ChannelWiseData {
            red: Array2::<f32>::zeros((self.height, self.width)),
            green: Array2::<f32>::zeros((self.height, self.width)),
            blue: Array2::<f32>::zeros((self.height, self.width)),
        };

        for (layer, layer_bias) in layers {
            for (x, y, pixel) in layer.into_rgb32f().enumerate_pixels() {
                let x = x as usize;
                let y = y as usize;

                let [red, green, blue] = pixel.0;
                result.red[[y, x]] += red * layer_bias;
                result.green[[y, x]] += green * layer_bias;
                result.blue[[y, x]] += blue * layer_bias;
            }
        }

        for (x, y, pixel) in self.residue.to_rgb32f().enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;

            let [red, green, blue] = pixel.0;
            result.red[[y, x]] += red * self.residue_bias;
            result.green[[y, x]] += green * self.residue_bias;
            result.blue[[y, x]] += blue * self.residue_bias;
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
            ImageBuffer::new(self.width as u32, self.height as u32);

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
