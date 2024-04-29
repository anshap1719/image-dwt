use convolve_image::kernel::{NonSeparableKernel, SeparableKernel};

#[derive(Copy, Clone)]
pub(crate) struct LinearInterpolationKernel(SeparableKernel<3>);

impl LinearInterpolationKernel {
    pub(crate) fn new() -> Self {
        Self(SeparableKernel::new([1. / 4., 1. / 2., 1. / 4.]))
    }
}

#[derive(Copy, Clone)]
pub(crate) struct LowScaleKernel(NonSeparableKernel<3>);

impl LowScaleKernel {
    #[allow(unused)]
    pub(crate) fn new() -> Self {
        Self(NonSeparableKernel::new([
            [1. / 16., 1. / 8., 1. / 16.],
            [1. / 8., 10., 1. / 8.],
            [1. / 16., 1. / 8., 1. / 16.],
        ]))
    }
}

#[derive(Copy, Clone)]
pub(crate) struct B3SplineKernel(SeparableKernel<5>);

impl B3SplineKernel {
    pub(crate) fn new() -> Self {
        Self(SeparableKernel::new([
            1. / 16.,
            1. / 4.,
            3. / 8.,
            1. / 4.,
            1. / 16.,
        ]))
    }
}

impl From<LinearInterpolationKernel> for SeparableKernel<3> {
    fn from(value: LinearInterpolationKernel) -> Self {
        value.0
    }
}

impl From<LowScaleKernel> for NonSeparableKernel<3> {
    fn from(value: LowScaleKernel) -> Self {
        value.0
    }
}

impl From<B3SplineKernel> for SeparableKernel<5> {
    fn from(value: B3SplineKernel) -> Self {
        value.0
    }
}

#[derive(Copy, Clone)]
pub enum Kernel {
    LinearInterpolationKernel,
    LowScaleKernel,
    B3SplineKernel,
}
