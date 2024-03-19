use image::{ImageBuffer, Rgb};
use ndarray::Array2;

pub struct WaveletLayerBuffer {
    pub(crate) data: Array2<f32>,
    pub(crate) pixel_scale: usize,
}

pub struct WaveletLayer {
    pub image_buffer: ImageBuffer<Rgb<f32>, Vec<f32>>,
    pub pixel_scale: Option<usize>,
}
