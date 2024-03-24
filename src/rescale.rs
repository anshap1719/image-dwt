use image::DynamicImage;
use ndarray::Array2;

use crate::aggregate::Aggregate;
use crate::transform::{ChannelWiseData, Scale};

pub trait RescalableImage {
    fn get_scaling_parameters(&self) -> [Scale; 3];
}

impl RescalableImage for DynamicImage {
    fn get_scaling_parameters(&self) -> [Scale; 3] {
        let (width, height) = (self.width() as usize, self.height() as usize);

        let mut result = ChannelWiseData {
            red: Array2::<f32>::zeros((height, width)),
            green: Array2::<f32>::zeros((height, width)),
            blue: Array2::<f32>::zeros((height, width)),
        };

        for (x, y, pixel) in self.to_rgb32f().enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;

            let [red, green, blue] = pixel.0;
            result.red[[y, x]] += red;
            result.green[[y, x]] += green;
            result.blue[[y, x]] += blue;
        }

        [
            Scale::new(result.red.min(), result.red.max()),
            Scale::new(result.green.min(), result.green.max()),
            Scale::new(result.blue.min(), result.blue.max()),
        ]
    }
}
