use asterusk::prelude::*;

use ndarray::prelude::*;

fn main() {
    let path = "examples/gray.jpeg";
    let array = open_grayimage_and_convert_to_ndarray2(&path).expect("Couldn't open the image");

    let kernel = array![[1., 0., 1.], [0., -6., 0.], [1., 0., 1.]];
    let conv_op = ConvOp::default(&kernel).stride((1, 1)).build();
    let result = conv_op.sum_convolution(&array);

    let img = ndarray2_to_gray_image(&result);
    img.save("examples/result.png")
        .expect("Couldn't save the image");
}
