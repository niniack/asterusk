use fastblur::gaussian_blur_asymmetric_single_channel;
use ndarray::prelude::*;
use std::iter::FromIterator;

// WIP
pub fn canny(input: Array2<u8>, low_threshold: f32, high_threshold: f32) {
    // Credits:
    // https://en.wikipedia.org/wiki/Canny_edge_detector
    // https://github.com/image-rs/imageproc/blob/bd4919d01b0c6d562ea1bd1a812b6b278e856bdc/src/edges.rs

    assert!(low_threshold <= high_threshold);

    // Apply Gaussian blur
    let shape = input.dim();
    let mut blurred = Array::from_iter(input.iter().cloned()).to_vec();
    gaussian_blur_asymmetric_single_channel(&mut blurred, shape.1, shape.0, 5.0, 5.0);
    // utils::write_gray_image("blur.ppm", &data, shape.1 as usize, shape.0 as usize).unwrap();

    // Find intensity gradients
}
