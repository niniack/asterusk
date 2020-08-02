use ndarray::prelude::*;
use ndarray::stack;
use ndarray_stats::QuantileExt;

fn max_pool2D(mut array: Array2<f32>) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    while (array.ncols() % 2) != 0 {
        array = stack!(Axis(1), array, Array2::zeros((array.nrows(), 1)));
    }
    while (array.nrows() % 2) != 0 {
        array = stack!(Axis(0), array, Array2::zeros((1, array.ncols())));
    }

    println!("{:#?}", &array);

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
fn max_pool() {
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
