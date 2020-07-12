#[cfg(test)]
mod tests {
    use image::*;
    #[allow(unused_imports)]
    use ndarray::prelude::*;
    use ndarray_image::open_gray_image;
    use std::path::Path;

    use crate::fftnd::{fft2d, ifft2d};
    use crate::utils;
    use rustfft::num_complex::Complex;

    #[test]
    fn test_basic_neuron() {
        fn mac(inputs: Array1<f64>, weights: Array1<f64>, bias: f64) -> f64 {
            let output = inputs.dot(&weights) + bias;
            return output;
        }

        let inputs: Array1<f64> = array![1.0, 2.0, 3.0];
        let weights: Array1<f64> = array![0.2, 0.8, -0.5];
        let bias: f64 = 2.0;
        let output = mac(inputs, weights, bias);

        assert_eq!(output, 2.3);
    }

    #[test]
    fn test_load_gray_image() {
        let path = Path::new("examples/gray.jpeg");
        let image = match open_gray_image(&path) {
            Ok(image) => image,
            Err(error) => panic!("Problem opening the image file: {:?}", error),
        };

        assert_eq!(image.ndim(), 2);
    }

    fn assert_eq_vecs(a: &[Complex<f32>], b: &[Complex<f32>]) {
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).norm() < 0.1f32);
        }
    }

    #[test]
    fn test_fft() {
        let mut input: Array2<f32> = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let mut input_complex = utils::f32_to_complex(&mut input);
        let mut output: Array2<Complex<f32>> = Array::zeros((3, 3));
        let mut expected: Array2<Complex<f32>> = array![
            [
                Complex::new(45.0, 0.),
                Complex::new(-4.5, 2.59807621),
                Complex::new(-4.5, -2.59807621)
            ],
            [
                Complex::new(-13.5, 7.79422863),
                Complex::new(0.0, 0.),
                Complex::new(0.0, 0.)
            ],
            [
                Complex::new(-13.5, -7.79422863),
                Complex::new(0.0, 0.),
                Complex::new(0.0, 0.)
            ]
        ];
        fft2d(&mut input_complex, &mut output);
        assert_eq_vecs(
            output.as_slice_mut().unwrap(),
            expected.as_slice_mut().unwrap(),
        );
    }

    #[test]
    fn test_ifft() {
        let mut input: Array2<Complex<f32>> = array![
            [
                Complex::new(45.0, 0.),
                Complex::new(-4.5, 2.59807621),
                Complex::new(-4.5, -2.59807621)
            ],
            [
                Complex::new(-13.5, 7.79422863),
                Complex::new(0.0, 0.),
                Complex::new(0.0, 0.)
            ],
            [
                Complex::new(-13.5, -7.79422863),
                Complex::new(0.0, 0.),
                Complex::new(0.0, 0.)
            ]
        ];
        let mut output: Array2<Complex<f32>> = Array::zeros((3, 3));
        let mut expected: Array2<Complex<f32>> = array![
            [
                Complex::new(1., 0.),
                Complex::new(2., 0.),
                Complex::new(3., 0.)
            ],
            [
                Complex::new(4., 0.),
                Complex::new(5., 0.),
                Complex::new(6., 0.)
            ],
            [
                Complex::new(7., 0.),
                Complex::new(8., 0.),
                Complex::new(9., 0.)
            ]
        ];
        ifft2d(&mut input, &mut output);
        // assert_eq_vecs(output.as_slice_mut().unwrap(), expected.as_slice_mut().unwrap());
    }

    #[test]
    fn test_image_to_ndarray2() {
        let path = "examples/gray.jpeg";
        let array = utils::open_grayimage_and_convert_to_ndarray2(&path).unwrap();
        let img = utils::ndarray2_to_gray_image(&array);
        let save_path = "examples/result.png";
        img.save(&save_path)
            .expect("Coudln't save the result image");

        let orig = image::open(&path).unwrap();
        let saved = image::open(&save_path).unwrap();
        let (w, h) = orig.dimensions();
        for y in 0..h {
            for x in 0..w {
                assert!(orig.get_pixel(x, y) == saved.get_pixel(x, y));
            }
        }
    }
}
