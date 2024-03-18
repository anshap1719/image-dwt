use image_dwt::transform::a_trous_transform;

fn main() {
    let image = image::open("./sample.jpg").unwrap();

    a_trous_transform(&image, 5);
}
