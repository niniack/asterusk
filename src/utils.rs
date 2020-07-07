use ndarray::prelude::*;

pub fn flatten(input: Array2<u8>) -> Array1<u8> {
    let shape = input.dim();
    let mut output: Array1<u8> = Array::zeros((shape.0*shape.1).f());
    for y in 0..shape.0 {
        for x in 0..shape.1 {
            output[(shape.1 * y) + x] = input[[y,x]];
        }
    }
    return output;
}

// Adapted from:
// https://github.com/fschutt/fastblur/blob/master/src/utils.rs
pub fn write_gray_image<S>(filename: S, data: &Vec<u8>, width: usize, height: usize) -> Result<(), ::std::io::Error> where S: Into<String> {
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
