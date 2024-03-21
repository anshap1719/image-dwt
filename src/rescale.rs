use std::cmp::Ordering;

use image::{ImageBuffer, Rgb};

pub trait RescalableImage {
    fn rescale(&mut self);
    fn channel_wise_rescale(&mut self);
}

impl RescalableImage for ImageBuffer<Rgb<f32>, Vec<f32>> {
    fn rescale(&mut self) {
        let max = self
            .iter()
            .max_by(|x, y| {
                if x > y {
                    Ordering::Greater
                } else if y > x {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .copied()
            .unwrap_or(1.)
            .max(1.);

        let min = self
            .iter()
            .min_by(|x, y| {
                if x > y {
                    Ordering::Greater
                } else if y > x {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .copied()
            .unwrap_or(0.)
            .min(0.);

        let diff = max - min;

        for pixel in self.pixels_mut() {
            let [r, g, b] = pixel.0;
            *pixel = Rgb([(r - min) / diff, (g - min) / diff, (b - min) / diff]);
        }
    }

    fn channel_wise_rescale(&mut self) {
        let mut max_r: f32 = 1.;
        let mut max_g: f32 = 1.;
        let mut max_b: f32 = 1.;

        let mut min_r: f32 = 0.;
        let mut min_g: f32 = 0.;
        let mut min_b: f32 = 0.;

        for pixel in self.pixels() {
            let [r, g, b] = pixel.0;
            max_r = max_r.max(r);
            max_g = max_g.max(g);
            max_b = max_b.max(b);

            min_r = min_r.min(r);
            min_g = min_g.min(g);
            min_b = min_b.min(b);
        }

        let diff_r = max_r - min_r;
        let diff_g = max_g - min_g;
        let diff_b = max_b - min_b;

        for pixel in self.pixels_mut() {
            let [r, g, b] = pixel.0;
            *pixel = Rgb([
                (r - min_r) / diff_r,
                (g - min_g) / diff_g,
                (b - min_b) / diff_b,
            ]);
        }
    }
}

impl RescalableImage for ImageBuffer<Rgb<f64>, Vec<f64>> {
    fn rescale(&mut self) {
        let max = self
            .iter()
            .max_by(|x, y| {
                if x > y {
                    Ordering::Greater
                } else if y > x {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .copied()
            .unwrap_or(1.);

        let min = self
            .iter()
            .min_by(|x, y| {
                if x > y {
                    Ordering::Greater
                } else if y > x {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .copied()
            .unwrap_or(0.);

        let diff = max - min;

        for pixel in self.pixels_mut() {
            let [r, g, b] = pixel.0;
            *pixel = Rgb([(r - min) / diff, (g - min) / diff, (b - min) / diff]);
        }
    }

    fn channel_wise_rescale(&mut self) {
        let mut max_r: f64 = 1.;
        let mut max_g: f64 = 1.;
        let mut max_b: f64 = 1.;

        let mut min_r: f64 = 0.;
        let mut min_g: f64 = 0.;
        let mut min_b: f64 = 0.;

        for pixel in self.pixels() {
            let [r, g, b] = pixel.0;
            max_r = max_r.max(r);
            max_g = max_g.max(g);
            max_b = max_b.max(b);

            min_r = min_r.min(r);
            min_g = min_g.min(g);
            min_b = min_b.min(b);
        }

        let diff_r = max_r - min_r;
        let diff_g = max_g - min_g;
        let diff_b = max_b - min_b;

        for pixel in self.pixels_mut() {
            let [r, g, b] = pixel.0;
            *pixel = Rgb([
                (r - min_r) / diff_r,
                (g - min_g) / diff_g,
                (b - min_b) / diff_b,
            ]);
        }
    }
}
