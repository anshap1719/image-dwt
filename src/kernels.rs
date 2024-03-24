use ndarray::Array2;

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

    fn compute_extended_index(
        &self,
        x: usize,
        y: usize,
        x_distance: isize,
        y_distance: isize,
        data: &Array2<f32>,
    ) -> [usize; 2] {
        let kernel_size = self.size() as isize;
        let kernel_padding = kernel_size / 2;
        let (max_y, max_x) = data.dim();

        let mut x = x as isize + x_distance;
        let mut y = y as isize + y_distance;

        if x < 0 {
            x = -x;
        } else if x > max_x as isize - kernel_padding {
            let overshot_distance = x - max_x as isize + kernel_padding;
            x = max_x as isize - overshot_distance;
        }

        if y < 0 {
            y = -y;
        } else if y > max_y as isize - kernel_padding {
            let overshot_distance = y - max_y as isize + kernel_padding;
            y = max_y as isize - overshot_distance;
        }

        [y as usize, x as usize]
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
