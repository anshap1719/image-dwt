use std::ops::Index;

pub trait Kernel
where
    Self: Copy,
    <Self::Values as Index<usize>>::Output: Index<usize, Output = f32> + IntoIterator,
{
    type Values: Index<usize> + IntoIterator;

    fn values(&self) -> Self::Values;

    fn size(&self) -> usize;

    fn index(&self, index: [usize; 2]) -> f32 {
        if index[0] > self.size() || index[1] > self.size() {
            panic!("Index out of bounds");
        }

        self.values()[index[0]][index[1]]
    }
}

#[derive(Copy, Clone)]
pub struct LinearInterpolationKernel;

impl Kernel for LinearInterpolationKernel {
    type Values = [[f32; 3]; 3];

    fn values(&self) -> Self::Values {
        [
            [1. / 16., 1. / 8., 1. / 16.],
            [1. / 8., 1. / 4., 1. / 8.],
            [1. / 16., 1. / 8., 1. / 16.],
        ]
    }

    fn size(&self) -> usize {
        3
    }
}

#[derive(Copy, Clone)]
pub struct LowScaleKernel;

impl Kernel for LowScaleKernel {
    type Values = [[f32; 3]; 3];

    fn values(&self) -> Self::Values {
        [
            [1. / 16., 1. / 8., 1. / 16.],
            [1. / 8., 10., 1. / 8.],
            [1. / 16., 1. / 8., 1. / 16.],
        ]
    }

    fn size(&self) -> usize {
        3
    }
}

#[derive(Copy, Clone)]
pub struct B3SplineKernel;

impl Kernel for B3SplineKernel {
    type Values = [[f32; 5]; 5];

    fn values(&self) -> Self::Values {
        [
            [1. / 256., 1. / 64., 3. / 128., 1. / 64., 1. / 256.],
            [1. / 64., 1. / 16., 3. / 32., 1. / 16., 1. / 64.],
            [3. / 128., 3. / 32., 9. / 64., 3. / 32., 3. / 128.],
            [1. / 64., 1. / 16., 3. / 32., 1. / 16., 1. / 64.],
            [1. / 256., 1. / 64., 3. / 128., 1. / 64., 1. / 256.],
        ]
    }

    fn size(&self) -> usize {
        5
    }
}
