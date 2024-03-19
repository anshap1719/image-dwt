use ndarray::Array2;

pub(crate) trait ExtendedIndex {
    fn compute_extended_index(
        &self,
        x: usize,
        y: usize,
        x_distance: isize,
        y_distance: isize,
    ) -> [usize; 2];
}

impl ExtendedIndex for Array2<f32> {
    fn compute_extended_index(
        &self,
        x: usize,
        y: usize,
        x_distance: isize,
        y_distance: isize,
    ) -> [usize; 2] {
        let (max_y, max_x) = self.dim();

        let mut x = x as isize + x_distance;
        let mut y = y as isize + y_distance;

        if x < 0 {
            x = -x;
        } else if x > max_x as isize - 1 {
            let overshot_distance = x - max_x as isize + 1;
            x = max_x as isize - overshot_distance;
        }

        if y < 0 {
            y = -y;
        } else if y > max_y as isize - 1 {
            let overshot_distance = y - max_y as isize + 1;
            y = max_y as isize - overshot_distance;
        }

        [y as usize, x as usize]
    }
}
