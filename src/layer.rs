use image::{ImageBuffer, Luma, Rgb};
use ndarray::{Array2, Array3};

use crate::aggregate::Aggregate;

#[derive(Clone)]
pub enum WaveletLayerBuffer {
    Grayscale { data: Array2<f32> },
    Rgb { data: Array3<f32> },
    Raw { data: Array3<f32> },
}

impl Aggregate for WaveletLayerBuffer {
    fn min(&self) -> f32 {
        match self {
            WaveletLayerBuffer::Grayscale { data } => data.min(),
            WaveletLayerBuffer::Rgb { data } => data.min(),
            WaveletLayerBuffer::Raw { data } => data.min(),
        }
    }

    fn max(&self) -> f32 {
        match self {
            WaveletLayerBuffer::Grayscale { data } => data.max(),
            WaveletLayerBuffer::Rgb { data } => data.max(),
            WaveletLayerBuffer::Raw { data } => data.max(),
        }
    }
}

#[derive(Clone)]
pub enum WaveletLayerImageBuffer {
    Grayscale {
        buffer: ImageBuffer<Luma<f32>, Vec<f32>>,
    },
    Rgb {
        buffer: ImageBuffer<Rgb<f32>, Vec<f32>>,
    },
}

impl From<WaveletLayerBuffer> for WaveletLayerImageBuffer {
    fn from(value: WaveletLayerBuffer) -> Self {
        match value {
            WaveletLayerBuffer::Grayscale { data } => {
                let (height, width) = data.dim();

                let mut buffer: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::new(
                    u32::try_from(width)
                        .unwrap_or_else(|_| panic!("width cannot be larger than {}", u32::MAX)),
                    u32::try_from(height)
                        .unwrap_or_else(|_| panic!("height cannot be larger than {}", u32::MAX)),
                );

                for (x, y, pixel) in buffer.enumerate_pixels_mut() {
                    let intensity = data[(y as usize, x as usize)];
                    *pixel = Luma([intensity]);
                }

                WaveletLayerImageBuffer::Grayscale { buffer }
            }
            WaveletLayerBuffer::Rgb { data } => {
                let (height, width, _) = data.dim();

                let mut buffer: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::new(
                    u32::try_from(width)
                        .unwrap_or_else(|_| panic!("width cannot be larger than {}", u32::MAX)),
                    u32::try_from(height)
                        .unwrap_or_else(|_| panic!("width cannot be larger than {}", u32::MAX)),
                );

                for (x, y, pixel) in buffer.enumerate_pixels_mut() {
                    let red = data[(y as usize, x as usize, 0)];
                    let green = data[(y as usize, x as usize, 1)];
                    let blue = data[(y as usize, x as usize, 2)];

                    *pixel = Rgb([red, green, blue]);
                }

                WaveletLayerImageBuffer::Rgb { buffer }
            }
            WaveletLayerBuffer::Raw { .. } => unimplemented!(),
        }
    }
}

#[derive(Clone)]
pub struct WaveletLayer {
    pub buffer: WaveletLayerBuffer,
    pub pixel_scale: Option<usize>,
}

impl WaveletLayer {
    #[must_use]
    pub fn is_residue_layer(&self) -> bool {
        self.pixel_scale.is_none()
    }
}
