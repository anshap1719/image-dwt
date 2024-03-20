# À Trous Discrete Wavelet Transform (DWT) for image-rs

This project provides an implementation of the À Trous Discrete Wavelet Transform (DWT) algorithm for images. The À
Trous DWT is a technique used for signal and image processing, particularly for tasks such as denoising, compression,
and feature extraction.

## Overview

The À Trous DWT is a variation of the Discrete Wavelet Transform (DWT) that involves convolution with a filter bank. It
decomposes an image into different frequency sub-bands, allowing for analysis at multiple resolutions. This
implementation supports both forward and inverse transforms.

## Why

I'm trying to build a suite of tools in rust that facilitate image processing, primarily deep sky images and data.
Wavelet transform and multi-resolution analysis is a very widely used transform in these cases.

## Usage

```rust
fn forward_transform() {
    let image = image::open("./sample.jpg").unwrap();
    let transform = ATrousTransform::new(&image, 3, LinearInterpolationKernel);

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
}
```

```rust
fn inverse_transform() {
    let recomposed = ATrousRecompose::new(
        &image::open("residue.jpg").unwrap()
    )
        .recompose((0..3).map(|layer| image::open(format!("level{layer}.jpg")).unwrap()));

    recomposed.to_rgb8().save("recombined.jpg").unwrap()
}
```

## Installation

To use this library in your Rust project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
image_dwt = "0.1.0"