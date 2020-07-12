use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct ConvOp {
    kernel: Array2<f32>,
    padding: (usize, usize), // (x,y)
    stride: (usize, usize),  // (x,y)
}

impl ConvOp {
    pub fn default(kernel: &Array2<f32>) -> Self {
        ConvOp {
            kernel: kernel.clone(),
            padding: (0, 0),
            stride: (1, 1),
        }
    }

    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    pub fn build(self) -> ConvOp {
        ConvOp {
            kernel: self.kernel,
            padding: self.padding,
            stride: self.stride,
        }
    }

    pub fn sum_convolution(&self, input: &Array2<f32>) -> Array2<f32> {
        let (i_n, i_m) = (input.shape()[0], input.shape()[1]);
        let kernel = &self.kernel;
        let (k_n, k_m) = (kernel.shape()[0], kernel.shape()[1]);
        // println!("Kernel shape is: {:?}",(k_n,k_m));

        if self.stride == (1, 1) {
            let (o_n, o_m) = (i_n - k_n + self.stride.1, i_m - k_m + self.stride.0);
            // println!("Output shape is: {:?}",(o_n,o_m));
            let mut output: Array2<f32> = Array::zeros((o_n, o_m));

            // println!("{:#?}", output);
            for y in 0..o_n {
                for x in 0..o_m {
                    let input_subview = input.slice(s![y..(y + k_n), x..(x + k_m)]);
                    // println!("input_subview:\n{:?}",input_subview);
                    output[[y, x]] = (&input_subview * kernel).sum();
                }
            }
            output
        } else {
            panic!("Convolution for stride != 1 has not been implemented yet");
        }
    }
}
