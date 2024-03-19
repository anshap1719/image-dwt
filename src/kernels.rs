pub trait Kernel<const SIZE: usize>
where
    Self: Copy + Send + Sync,
{
    fn values(&self) -> [[f32; SIZE]; SIZE];

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

impl Kernel<3> for LinearInterpolationKernel {
    fn values(&self) -> [[f32; 3]; 3] {
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

impl Kernel<3> for LowScaleKernel {
    fn values(&self) -> [[f32; 3]; 3] {
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

impl Kernel<5> for B3SplineKernel {
    fn values(&self) -> [[f32; 5]; 5] {
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
