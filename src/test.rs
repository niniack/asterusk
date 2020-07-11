#[cfg(test)]
mod tests{
    #[allow(unused_imports)]

    use ndarray::prelude::*;
    use ndarray_image::open_gray_image;
    use std::path::Path;

    use rustfft::num_complex::Complex;
    use crate::fftnd::fft2d;
    use crate::utils;

    #[test]
    fn test_basic_neuron() {
        fn mac(inputs: Array1<f64>, weights: Array1<f64>, bias: f64) -> f64{
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

        assert_eq!(image.ndim(),2);
    }

    #[test]
    fn test_fft() {
        let mut input: Array2<f32> = array![[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]];
        let mut input_complex = utils::f32_to_complex(&mut input);
        let mut output: Array2<Complex<f32>> = Array::zeros((3,3));
        let expected: Array2<Complex<f32>> = array![[Complex::new( 45.0,  0.        ), Complex::new(-4.5, 2.59807621), Complex::new(-4.5, -2.59807621)],
                                                    [Complex::new(-13.5,  7.79422863), Complex::new( 0.0, 0.        ), Complex::new( 0.0,  0.        )],
                                                    [Complex::new(-13.5, -7.79422863), Complex::new( 0.0, 0.        ), Complex::new( 0.0,  0.        )]];
        fft2d(&mut input_complex, &mut output);
        assert_eq!(output, expected);
    }
}
