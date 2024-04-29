use image_dwt::recompose::{OutputLayer, RecomposableWaveletLayers};
use image_dwt::transform::ATrousTransform;
use image_dwt::Kernel;

fn main() {
    let image = image::open("./sample.jpg").unwrap();
    let transform = ATrousTransform::new(&image, 6, Kernel::B3SplineKernel);

    let recomposed = transform
        .into_iter()
        .skip(1)
        .filter(|item| item.pixel_scale.is_some_and(|scale| scale < 2))
        .recompose_into_image(
            image.width() as usize,
            image.height() as usize,
            OutputLayer::Rgb,
        );

    recomposed.to_rgb8().save("recombined.jpg").unwrap()
}
