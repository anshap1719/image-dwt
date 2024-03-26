use ndarray::{Array2, Array3};

pub trait Aggregate {
    fn min(&self) -> f32;
    fn max(&self) -> f32;
}

impl Aggregate for Array2<f32> {
    fn min(&self) -> f32 {
        *self
            .iter()
            .reduce(|current, previous| {
                if current < previous {
                    current
                } else {
                    previous
                }
            })
            .unwrap()
    }

    fn max(&self) -> f32 {
        *self
            .iter()
            .reduce(|current, previous| {
                if current > previous {
                    current
                } else {
                    previous
                }
            })
            .unwrap()
    }
}

impl Aggregate for Array3<f32> {
    fn min(&self) -> f32 {
        *self
            .iter()
            .reduce(|current, previous| {
                if current < previous {
                    current
                } else {
                    previous
                }
            })
            .unwrap()
    }

    fn max(&self) -> f32 {
        *self
            .iter()
            .reduce(|current, previous| {
                if current > previous {
                    current
                } else {
                    previous
                }
            })
            .unwrap()
    }
}
