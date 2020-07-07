use std::path::Path;
use ndarray_image::open_gray_image;

use asterusk::edge_detection::canny;

fn main() {
    let path = Path::new("examples/gray.jpeg");
    let image = match open_gray_image(&path) {
        Ok(image) => image,
        Err(error) => panic!("Problem opening the image file: {:?}", error),
    };
    canny(image, 10.0, 20.0);
}
