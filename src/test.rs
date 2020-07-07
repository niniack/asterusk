use ndarray::prelude::*;

#[test]
fn basic_neuron() {
    fn mac(inputs: Array1<f64>, weights: Array1<f64>, bias: f64) -> f64{
        let output = inputs.dot(&weights) + bias;
        println!("{}", output);
        return output;
    }

    let inputs: Array1<f64> = array![1.0, 2.0, 3.0];
    let weights: Array1<f64> = array![0.2, 0.8, -0.5];
    let bias: f64 = 2.0;
    let output = mac(inputs, weights, bias);
    assert_eq!(output, 2.3);
}
