use image::DynamicImage;

use image_dwt::kernels::LinearInterpolationKernel;
use image_dwt::transform::ATrousTransform;

fn main() {
    let image = image::open("./5bef9d1cc91f5635e4274f8df62f6906.jpg").unwrap();
    let transform = ATrousTransform::new(&image, 10, LinearInterpolationKernel);

    for layer in transform {
        let name = match layer.pixel_scale {
            Some(pixel_scale) => format!("level{pixel_scale}"),
            None => "residue".to_string(),
        };

        let image = DynamicImage::ImageRgb32F(layer.image_buffer);
        image.to_rgb8().save(format!("{name}.jpg")).unwrap();
    }

    // a_trous_transform(&image, 10);

    // let image = image::open("./residue.jpg").unwrap();
    // let (width, height) = image.dimensions();
    // let mut buffer = Array2::<f32>::zeros((height as usize, width as usize));
    //
    // for (x, y, pixel) in image.to_luma32f().enumerate_pixels() {
    //     buffer[[y as usize, x as usize]] = pixel.0[0];
    // }
    //
    // for level in 0..2 {
    //     let image = image::open(format!("./level{level}.jpg")).unwrap();
    //     let mut level_buffer =
    //         Array2::<f32>::zeros((image.height() as usize, image.width() as usize));
    //
    //     for (x, y, pixel) in image.to_luma32f().enumerate_pixels() {
    //         level_buffer[[y as usize, x as usize]] = pixel.0[0];
    //     }
    //
    //     buffer += &level_buffer;
    // }
    //
    // let min_pixel = buffer
    //     .iter()
    //     .reduce(|current, previous| {
    //         if current < previous {
    //             current
    //         } else {
    //             previous
    //         }
    //     })
    //     .unwrap();
    //
    // let max_pixel = buffer
    //     .iter()
    //     .reduce(|current, previous| {
    //         if current > previous {
    //             current
    //         } else {
    //             previous
    //         }
    //     })
    //     .unwrap();
    //
    // let mut result_img: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::new(width, height);
    //
    // let rescale_ratio = max_pixel - min_pixel;
    //
    // for (x, y, pixel) in result_img.enumerate_pixels_mut() {
    //     let val = buffer[(y as usize, x as usize)];
    //     let scaled_value = (val - min_pixel) / rescale_ratio;
    //     *pixel = Rgb([scaled_value, scaled_value, scaled_value]);
    // }
    // DynamicImage::ImageRgb32F(result_img)
    //     .to_rgb8()
    //     .save("recombined.jpg")
    //     .unwrap();
}
