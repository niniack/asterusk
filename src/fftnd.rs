use ndarray::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{FFTnum, FFTplanner};

pub fn fft<T: FFTnum>(input: &mut [Complex<T>], output: &mut [Complex<T>], inverse: bool) {
    let mut planner = FFTplanner::new(inverse);
    let len = input.len();
    let fft = planner.plan_fft(len);
    fft.process(input, output);
}

// using https://docs.rs/rustfft/3.0.1/rustfft/
// Credits:
// https://github.com/totem3/ofuton/blob/master/src/lib.rs

pub fn ifft2d(input: &mut Array2<Complex<f32>>, output: &mut Array2<Complex<f32>>) {
    let shape = input.dim();

    // Transpose the input
    input.swap_axes(0, 1);

    // Instantiate fft_cols_complex array to store 1D Col FFT
    let mut fft_cols_complex: Array2<Complex<f32>> = Array::zeros((shape.1, shape.0));

    // Send input rows for 1D FFT
    let mut output_col_iters = fft_cols_complex.genrows_mut().into_iter();

    for input_col_iter in input.genrows_mut() {
        let mut output_col_iter = output_col_iters.next().unwrap();
        fft(
            &mut input_col_iter.to_vec(),
            output_col_iter.as_slice_mut().unwrap(),
            true,
        );
    }

    // Transpose the fft_cols_complex array
    fft_cols_complex.swap_axes(0, 1);

    // Send fft_cols_complex for 1D FFT
    let mut output_row_iters = output.genrows_mut().into_iter();

    for input_row_iter in fft_cols_complex.genrows_mut() {
        let mut output_row_iter = output_row_iters.next().unwrap();
        let mut out = vec![Zero::zero(); output_row_iter.len()];
        fft(&mut input_row_iter.to_vec(), &mut out, true);
        for i in 0..output_row_iter.len() {
            output_row_iter[i] = out.remove(0);
        }
    }
}

pub fn fft2d(input: &mut Array2<Complex<f32>>, output: &mut Array2<Complex<f32>>) {
    let shape = input.dim();

    // Instantiate fft_rows_complex array to store 1D Row FFT
    let mut fft_rows_complex: Array2<Complex<f32>> = Array::zeros((shape.0, shape.1));

    // Send input rows for 1D FFT
    let mut output_row_iters = fft_rows_complex.genrows_mut().into_iter();

    for mut input_row_iter in input.genrows_mut() {
        let mut output_row_iter = output_row_iters.next().unwrap();
        fft(
            input_row_iter.as_slice_mut().unwrap(),
            output_row_iter.as_slice_mut().unwrap(),
            false,
        );
    }

    // Transpose the fft_rows_complex
    // For now must transpose & use genrows_mut instead of directly using gencolumns_mut because of
    // layout issues in how the view is returned. Cannot unwrap the result
    fft_rows_complex.swap_axes(0, 1);
    output.swap_axes(0, 1);

    // Send fft_rows_complex columns for 1D FFT
    let mut output_col_iters = output.genrows_mut().into_iter();

    for input_col_iter in fft_rows_complex.genrows_mut() {
        let mut output_col_iter = output_col_iters.next().unwrap();
        let mut out = vec![Zero::zero(); output_col_iter.len()];
        fft(&mut input_col_iter.to_vec(), &mut out, false);
        for i in 0..output_col_iter.len() {
            output_col_iter[i] = out.remove(0);
        }
    }

    // Transpose output column to get into original shape
    output.swap_axes(0, 1);
}
