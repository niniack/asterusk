use ndarray::prelude::*;
use rustfft::num_complex::Complex;

use image::*;
use std::iter::FromIterator;

// Adapted from:
// https://github.com/fschutt/fastblur/blob/master/src/utils.rs
pub fn write_gray_image<S>(
    filename: S,
    data: &Vec<u8>,
    width: usize,
    height: usize,
) -> Result<(), ::std::io::Error>
where
    S: Into<String>,
{
    use std::fs::File;
    use std::io::BufWriter;
    use std::io::Write;

    let mut file = BufWriter::new(File::create(filename.into())?);
    let header = format!("P5\n{}\n{}\n{}\n", width, height, 255);

    file.write(header.as_bytes())?;

    for px in data {
        file.write(&px.to_be_bytes())?;
    }

    Ok(())
}

pub fn f32_to_complex(input: &Array2<f32>) -> Array2<Complex<f32>> {
    let shape = input.dim();
    let mut output: Array2<Complex<f32>> = Array::zeros((shape.0, shape.1));
    for y in 0..shape.0 {
        for x in 0..shape.1 {
            output[[y, x]] = Complex::from(input[[y, x]]);
        }
    }
    return output;
}

use std::convert::TryInto;

pub fn open_grayimage_and_convert_to_ndarray2(path: &str) -> Result<Array2<f32>, ImageError> {
    let img = image::open(&path)?.to_luma();

    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    println!("img dimensions: ({},{})", w, h);

    let mut array = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            array[[y, x]] = img.get_pixel(x as u32, y as u32)[0] as f32;
        }
    }

    Ok(array)
}

pub fn ndarray2_to_gray_image(array: &Array2<f32>) -> GrayImage {
    assert!(array.is_standard_layout());

    let (w, h) = array.dim();
    println!("Array (width,height): ({},{})", w, h);
    let mut v = Array::from_iter(array.iter().cloned()).to_vec();

    let mut min = 255.;
    let mut max = 0.;
    for elem in &v {
        if elem < &min {
            min = *elem;
        } else if elem > &max {
            max = *elem;
        }
    }

    let raw = v
        .iter()
        .map(|x| (((*x - min) / (max - min)) * 255.) as u8)
        .collect();
    for pixel in &raw {
        // println!("pixel: {}",pixel);
        if pixel > &255 || pixel < &0 {
            panic!("We're not supposed to have a value this size");
        }
    }

    let img: GrayImage = GrayImage::from_raw(h as u32, w as u32, raw).expect("Couldn't convert");
    img
}

pub fn open_image_and_convert_to_ndarray3(path: &str) -> Result<Array3<f32>, ImageError> {
    let img = image::open(&path)?;

    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    println!("img dimensions: ({},{})", w, h);

    let mut array = Array3::<f32>::zeros((3, w, h));
    for y in 0..h {
        for x in 0..w {
            array[[0, x, y]] = img.get_pixel(x as u32, y as u32)[0] as f32;
            array[[1, x, y]] = img.get_pixel(x as u32, y as u32)[1] as f32;
            array[[2, x, y]] = img.get_pixel(x as u32, y as u32)[2] as f32;
        }
    }

    Ok(array)
}

pub fn ndarray3_to_rgb_image(array: Array3<f32>) -> Result<RgbImage, ImageError> {
    let dims = array.dim();
    let (w, h) = (dims.1, dims.2);

    let mut img: RgbImage = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let (x, y) = (x as usize, y as usize);
            img.put_pixel(
                x as u32,
                y as u32,
                Rgb([
                    array[[0, x, y]] as u8,
                    array[[1, x, y]] as u8,
                    array[[2, x, y]] as u8,
                ]),
            );
        }
    }

    Ok(img)
}
