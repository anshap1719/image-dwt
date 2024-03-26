use image::{ImageBuffer, Rgb};
use ndarray::Array2;

pub struct WaveletLayerBuffer {
    pub(crate) data: Array2<f32>,
}

pub struct WaveletLayer {
    pub image_buffer: ImageBuffer<Rgb<f32>, Vec<f32>>,
    pub pixel_scale: Option<usize>,
}

impl WaveletLayer {
    pub fn is_residue_layer(&self) -> bool {
        self.pixel_scale.is_none()
    }
}
