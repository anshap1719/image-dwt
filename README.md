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
fn remove_large_scale_structures() {
    let image = image::open("./sample.jpg").unwrap();
    let transform = ATrousTransform::new(&image, 6, B3SplineKernel);

    let recomposed = transform
        .into_iter()
        // Skip pixel scale 0 layer for noise removal
        .skip(1)
        // Only take layers where pixel scale is less than 2
        .filter(|item| item.pixel_scale.is_some_and(|scale| scale < 2))
        // Recompose processed layers into final image
        .recompose_into_image(image.width() as usize, image.height() as usize);

    recomposed.to_rgb8().save("recombined.jpg").unwrap()
}
```

## Installation

To use this library in your Rust project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
image_dwt = "0.3.2"