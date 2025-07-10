use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use ndarray::{Array2, Array3};

use crate::aggregate::Aggregate;
use crate::layer::{WaveletLayer, WaveletLayerBuffer};

#[derive(Copy, Clone)]
pub enum OutputLayer {
    Grayscale,
    Rgb,
}

impl OutputLayer {
    fn to_num_channels(self) -> usize {
        match self {
            OutputLayer::Grayscale => 1,
            OutputLayer::Rgb => 3,
        }
    }
}

pub trait RecomposableWaveletLayers: Iterator<Item = WaveletLayer> {
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_lossless
    )]
    fn recompose_into_image(
        self,
        width: usize,
        height: usize,
        output_layer: OutputLayer,
    ) -> DynamicImage
    where
        Self: Sized,
    {
        match output_layer {
            OutputLayer::Grayscale => {
                let mut result = Array2::<f32>::zeros((height, width));

                for layer in self {
                    match layer.buffer {
                        WaveletLayerBuffer::Grayscale { data } => {
                            result += &data;
                        }
                        WaveletLayerBuffer::Rgb { .. } => {
                            panic!("Cannot create Grayscale output from RGB input")
                        }
                        WaveletLayerBuffer::Raw { .. } => {
                            panic!("Cannot create Grayscale output from Raw input")
                        }
                    }
                }

                let min_pixel = result.min();
                let max_pixel = result.max();

                let mut result_img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(
                    u32::try_from(width)
                        .unwrap_or_else(|_| panic!("width cannot be larger than {}", u32::MAX)),
                    u32::try_from(height)
                        .unwrap_or_else(|_| panic!("height cannot be larger than {}", u32::MAX)),
                );

                let rescale_ratio = max_pixel - min_pixel;

                for (x, y, pixel) in result_img.enumerate_pixels_mut() {
                    let intensity = result[(y as usize, x as usize)];

                    *pixel =
                        Luma([((intensity - min_pixel) / rescale_ratio * u16::MAX as f32) as u16]);
                }

                DynamicImage::ImageLuma16(result_img)
            }
            OutputLayer::Rgb => {
                let mut result =
                    Array3::<f32>::zeros((height, width, output_layer.to_num_channels()));

                for layer in self {
                    match layer.buffer {
                        WaveletLayerBuffer::Rgb { data } => {
                            result += &data;
                        }
                        WaveletLayerBuffer::Raw { .. } => {
                            panic!("Cannot create RGB output from RAW input")
                        }
                        WaveletLayerBuffer::Grayscale { .. } => {
                            panic!("Cannot create RGB output from Grayscale input")
                        }
                    }
                }

                let min_pixel = result.min();
                let max_pixel = result.max();

                let mut result_img: ImageBuffer<Rgb<f32>, Vec<f32>> =
                    ImageBuffer::new(width as u32, height as u32);

                let rescale_ratio = max_pixel - min_pixel;

                for (x, y, pixel) in result_img.enumerate_pixels_mut() {
                    let red = result[(y as usize, x as usize, 0)];
                    let green = result[(y as usize, x as usize, 1)];
                    let blue = result[(y as usize, x as usize, 2)];

                    let scaled_red = (red - min_pixel) / rescale_ratio;
                    let scaled_green = (green - min_pixel) / rescale_ratio;
                    let scaled_blue = (blue - min_pixel) / rescale_ratio;

                    *pixel = Rgb([scaled_red, scaled_green, scaled_blue]);
                }

                DynamicImage::ImageRgb32F(result_img)
            }
        }
    }
    fn recompose_into_raw(self, width: usize, height: usize) -> Array3<f32>
    where
        Self: Sized,
    {
        let mut result = Array3::<f32>::zeros((height, width, 3));

        for layer in self {
            match layer.buffer {
                WaveletLayerBuffer::Raw { data } => {
                    result += &data;
                }
                WaveletLayerBuffer::Rgb { .. } => {
                    panic!("Cannot create RGB output from RAW input")
                }
                WaveletLayerBuffer::Grayscale { .. } => {
                    panic!("Cannot create RGB output from Grayscale input")
                }
            }
        }

        result
    }
}

impl<T> RecomposableWaveletLayers for T where T: Iterator<Item = WaveletLayer> {}
