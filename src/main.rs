use image_dwt::recompose::{OutputLayer, RecomposableWaveletLayers};
use image_dwt::transform::ATrousTransform;
use image_dwt::Kernel;

fn main() {
    let image = image::open("./DSC09361.tif").expect("Failed to load image");
    let transform = ATrousTransform::new(&image, 10, Kernel::B3SplineKernel);

    let recomposed = transform.into_iter().recompose_into_image(
        image.width() as usize,
        image.height() as usize,
        OutputLayer::Rgb,
    );

    recomposed
        .to_rgb8()
        .save("recombined.jpg")
        .expect("Failed to save image");
}
