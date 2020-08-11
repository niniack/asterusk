use ndarray::prelude::*;
use ndarray::stack;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;

pub fn max_pool2D(mut array: Array2<f32>) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    while (array.ncols() % 2) != 0 {
        array = stack!(Axis(1), array, Array2::zeros((array.nrows(), 1)));
    }
    while (array.nrows() % 2) != 0 {
        array = stack!(Axis(0), array, Array2::zeros((1, array.ncols())));
    }

    let pooled_cols = array.ncols() / 2;
    let pooled_rows = array.nrows() / 2;

    let mut pooled_array = Array2::zeros((pooled_rows, pooled_cols));

    for j in 0..pooled_cols {
        for i in 0..pooled_rows {
            let y_start = j * 2;
            let x_start = i * 2;
            pooled_array[[i, j]] =
                *QuantileExt::max(&array.slice(s![x_start..x_start + 2, y_start..y_start + 2]))?;
        }
    }

    Ok(pooled_array)
}

#[test]
fn test_max_pool2D() {
    let array: Array2<f32> = array![
        [1., 2., 1., 4.],
        [3., 8., 1., 7.],
        [6., 2., 1., 4.],
        [9., 0., 6., 2.],
    ];

    println!("{:#?}", array);

    let pooled_array = max_pool2D(array).unwrap();

    println!("{:#?}", pooled_array);
}

pub fn max_pool3D(mut array: Array3<f32>) -> Result<(Array3<f32>), Box<dyn std::error::Error>> {
    let dims = array.dim();
    let mut pooled_array: Array3<f32> = Array::zeros((dims.0, dims.1 / 2, dims.2 / 2));

    let mut a: Vec<Array2<f32>> = Vec::new();
    for i in 0..dims.0 {
        a.push(array.slice_mut(s![i, .., ..]).to_owned());
        a[i] = max_pool2D(a[i].clone())?;

        for y in 0..pooled_array.dim().2 {
            for x in 0..pooled_array.dim().1 {
                pooled_array[[i, x, y]] = a[i][[x, y]];
            }
        }
    }

    Ok(pooled_array)
}

#[test]
fn test_max_pool_3D() {
    let mut array: Array3<f32> = Array::random((3, 4, 6), Uniform::new(0., 10.));
    println!("3D original array:\n{:#.1?}", array);
    let pooled_array = max_pool3D(array).expect("Error pooling 3D array");
    println!("3D pooled_array:\n{:#.1?}", pooled_array);
}

use crate::utils::*;

#[test]
fn test_max_pool_3D_rgb_image() {
    let array =
        open_image_and_convert_to_ndarray3("examples/ferris_ml.png").expect("Couldn't open image");

    let pooled_array = max_pool3D(array).expect("Couldn't run max_pool3D operation");

    let img = ndarray3_to_rgb_image(pooled_array).expect("Couldn't convert the Array3 to an RGB image");

    img.save("examples/rgb_pooled_result.png").expect("Couldn't save max pooled RGB image");

}
