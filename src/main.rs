use image::DynamicImage;

use image_dwt::kernels::B3SplineKernel;
use image_dwt::recompose::ATrousRecompose;
use image_dwt::transform::ATrousTransform;

fn main() {
    let image = image::open("./Dis.jpg").unwrap();
    let transform = ATrousTransform::new(&image, 6, B3SplineKernel);

    for layer in transform {
        let name = match layer.pixel_scale {
            Some(pixel_scale) => {
                format!("level{pixel_scale}")
            }
            None => "residue".to_string(),
        };

        let image = DynamicImage::ImageRgb32F(layer.image_buffer);
        image.to_rgb8().save(format!("{name}.jpg")).unwrap();
    }

    let recomposed = ATrousRecompose::new(&image::open("./residue.jpg").unwrap())
        .recompose((0..6).map(|layer| image::open(format!("level{layer}.jpg")).unwrap()));

    recomposed.to_rgb8().save("recombined1.jpg").unwrap()
}
