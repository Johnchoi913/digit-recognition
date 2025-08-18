use std::f64::{self};

use crate::generic_lib::HasPixels;
use rand::{prelude::*, rng};

#[derive(Clone)]
pub struct NeuralNetwork<T> {
    data: Vec<Vec<T>>,
    labels: Vec<Vec<usize>>,
    num_hidden_layer: usize,
    activations: Vec<Vec<f64>>,
    z_values: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
}

impl<T> NeuralNetwork<T>
where
    T: HasPixels,
    T: Clone,
{
    pub fn new(
        t: Vec<Vec<T>>,
        labels: Vec<Vec<usize>>,
        num_hidden_layer: usize,
        num_hidden_node: Vec<usize>,
    ) -> Self {
        let num_input = t[0][0].get_size();
        let num_output = t[0][0].get_num_output();
        let data = t;
        let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut biases: Vec<Vec<f64>> = Vec::new();

        let mut layer0 = vec![vec![0.0; num_hidden_node[0]]; num_input];

        for layer in layer0.iter_mut() {
            for node in layer.iter_mut() {
                *node = rand::rng().random_range(-0.1..0.1);
            }
        }

        weights.push(layer0);

        for x in 1..num_hidden_layer {
            let mut hidden = vec![vec![0.0; num_hidden_node[x]]; num_hidden_node[x - 1]];
            for row in hidden.iter_mut() {
                for node in row.iter_mut() {
                    *node = rand::rng().random_range(-0.1..0.1);
                }
            }
            weights.push(hidden);
        }

        let mut last = vec![vec![0.0; num_output]; num_hidden_node[num_hidden_layer - 1]];
        for row in last.iter_mut() {
            for node in row.iter_mut() {
                *node = rand::rng().random_range(-0.1..0.1);
            }
        }
        weights.push(last);

        biases.push(Vec::new());
        for _ in 0..num_hidden_node[0] {
            biases[0].push(rand::rng().random_range(-0.1..0.1));
        }

        for x in 0..num_hidden_layer - 1 {
            biases.push(Vec::new());
            for _ in 0..num_hidden_node[x + 1] {
                biases[x + 1].push(rand::rng().random_range(-0.1..0.1));
            }
        }

        biases.push(Vec::new());
        for _ in 0..num_output {
            biases[num_hidden_layer].push(rand::rng().random_range(-0.1..0.1));
        }

        /* Nodes in a hidden layer
        - Should be upper bound by :
            # of samples / (constant * (# of input + # of output) )
        - Should be around :
            \sqrt{# of input + # of output}
        */

        /*
        Dig Upper = 1000 / (2 * (28 * 29 + 10)) = 0.6
        Dig Approx = \sqrt{(28 * 29) + 10} = 28.7
        Digits do not benefit much from hidden layer

        Face Upper = 150 / (2 * (70 * 61 + 2)) = 0
        Face Approx = \sqrt{(70 * 61) + 2} = 65.4
        Faces do not benefit much from hidden layer
        */

        NeuralNetwork {
            data,
            labels,
            num_hidden_layer,
            activations: Vec::new(),
            z_values: Vec::new(),
            weights,
            biases,
        }
    }

    pub fn start(&mut self) {
        let mut rng = rng();

        let len = self.data[0].len();

        let copy = self.clone();

        let mut pair: Vec<(&T, &usize)> = copy.data[0].iter().zip(copy.labels[0].iter()).collect();

        for epoch in 0..10 {
            let training_amount = 0.1 * f64::from(epoch + 1) * len as f64;
            let training_amount = training_amount as usize;
            let (shuffled, _) = pair.partial_shuffle(&mut rng, training_amount);
            for (image, label) in shuffled.iter() {
                let predict = self.predict(image);
                self.backpropagation(&predict, label);
            }

            let (cnt, len) = self.validate();
            println!("{} {}", cnt, len);
        }
    }

    fn validate(&mut self) -> (usize, usize) {
        let mut count = 0;

        for x in 0..self.data[1].len() {
            if self
                .predict(&self.data[1][x].clone())
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|x| x.0)
                .unwrap()
                == self.labels[1][x]
            {
                count += 1;
            }
        }

        (count, self.data[2].len())
    }

    fn predict(&mut self, t: &T) -> Vec<f64> {
        let input = t.flatten();

        self.activations.push(input.clone());

        let mut layer = input;

        for x in 0..self.num_hidden_layer + 1 {
            let tmp = matrix_multiply(&[layer.clone()], &self.weights[x]);
            layer = add_vec(&tmp.first().unwrap().clone(), &self.biases[x]);

            self.z_values.push(layer.clone());

            if x < self.num_hidden_layer + 1 {
                for z in &mut layer {
                    *z = relu(*z);
                }
            } else {
                layer = softmax(layer);
            }

            self.activations.push(layer.clone());
        }

        layer
    }

    fn backpropagation(&mut self, prediction: &[f64], label: &usize) {
        let learning_rate = 0.01;

        let mut out_grad = prediction.to_owned();
        out_grad[*label] -= 1.0;

        let mut layer_grads = vec![out_grad];

        for i in (0..self.num_hidden_layer).rev() {
            let curr_layer_idx = i + 1;
            let next_grad = &layer_grads[0];

            let mut transposed_weights = vec![
                vec![0.0; self.weights[curr_layer_idx].len()];
                self.weights[curr_layer_idx][0].len()
            ];

            for (j, row) in transposed_weights.iter_mut().enumerate() {
                for (k, node) in row.iter_mut().enumerate() {
                    *node = self.weights[curr_layer_idx][k][j];
                }
            }

            let backprop_error =
                matrix_multiply(&[next_grad.clone()], &transposed_weights)[0].clone();

            let mut curr_grad = vec![0.0; backprop_error.len()];
            for j in 0..backprop_error.len() {
                curr_grad[j] = backprop_error[j] * relu_derivative(self.z_values[i][j]);
            }

            layer_grads.insert(0, curr_grad);
        }

        for layer in 0..=self.num_hidden_layer {
            let grad_idx = layer;

            let layer_input = if layer == 0 {
                &self.activations[0]
            } else {
                &self.activations[layer]
            };

            for i in 0..self.weights[layer].len() {
                for j in 0..self.weights[layer][i].len() {
                    if i < layer_input.len() && j < layer_grads[grad_idx].len() {
                        let delta = layer_input[i] * layer_grads[grad_idx][j];
                        self.weights[layer][i][j] -= learning_rate * delta;
                    }
                }
            }
            for j in 0..self.biases[layer].len() {
                if j < layer_grads[grad_idx].len() {
                    self.biases[layer][j] -= learning_rate * layer_grads[grad_idx][j];
                }
            }
        }

        self.activations.clear();
        self.z_values.clear();
    }
}
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        return 1.0;
    }
    0.0
}

fn softmax(outputs: Vec<f64>) -> Vec<f64> {
    let max_val = *outputs
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let exp_outputs: Vec<f64> = outputs.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp_outputs.iter().sum();

    exp_outputs.iter().map(|&x| x / sum).collect()
}

fn matrix_multiply(layer: &[Vec<f64>], weight: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if layer[0].len() != weight.len() {
        eprintln!(
            "Dot Product length of args differ: layer {} does not equal weight {}",
            layer[0].len(),
            weight.len()
        );
        panic!();
    }

    let mut result = vec![vec![0.0; weight[0].len()]; layer.len()];

    for i in 0..layer.len() {
        for j in 0..weight[0].len() {
            for k in 0..weight.len() {
                result[i][j] += layer[i][k] * weight[k][j];
            }
        }
    }

    result
}

fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() != b.len() {
        eprintln!(
            "Add_Vec Dimension Mismatch, {} does not equal {}",
            a.len(),
            b.len()
        );
        panic!();
    }

    a.iter().zip(b).map(|x| x.0 + x.1).collect()
}
