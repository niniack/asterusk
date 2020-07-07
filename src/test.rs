#[allow(unused_imports)]
use ndarray::prelude::*;
use ndarray_image::open_gray_image;
use std::path::Path;

#[test]
fn basic_neuron() {
    fn mac(inputs: Array1<f64>, weights: Array1<f64>, bias: f64) -> f64{
        let output = inputs.dot(&weights) + bias;
        println!("{}", output);
        return output;
    }

    let inputs: Array1<f64> = array![1.0, 2.0, 3.0];
    let weights: Array1<f64> = array![0.2, 0.8, -0.5];
    let bias: f64 = 2.0;
    let output = mac(inputs, weights, bias);
    assert_eq!(output, 2.3);
}

#[test]
fn load_gray_image() {
    let path = Path::new("examples/gray.jpeg");
    let image = match open_gray_image(&path) {
        Ok(image) => image,
        Err(error) => panic!("Problem opening the image file: {:?}", error),
    };
    assert_eq!(image.ndim(),2);
}
