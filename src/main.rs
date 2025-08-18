mod digit;
mod generic_lib;
mod neural_network;

use std::fs;

use digit::Digit;
use generic_lib::DIGIT_HEIGHT;
use neural_network::NeuralNetwork;

fn main() {
    let digit_vec: Vec<String> = fs::read_to_string("./data/digitdata/trainingimages")
        .unwrap()
        .lines()
        .collect::<Vec<&str>>()
        .chunks(DIGIT_HEIGHT)
        .map(|chunk| chunk.join("\n"))
        .collect();

    let digit_training_img: Vec<digit::Digit> = digit_vec.into_iter().map(Digit::new).collect();

    let digit_vec: Vec<String> = fs::read_to_string("./data/digitdata/testimages")
        .unwrap()
        .lines()
        .collect::<Vec<&str>>()
        .chunks(DIGIT_HEIGHT)
        .map(|chunk| chunk.join("\n"))
        .collect();

    let digit_testing_img: Vec<digit::Digit> = digit_vec.into_iter().map(Digit::new).collect();

    let digit_vec: Vec<String> = fs::read_to_string("./data/digitdata/validationimages")
        .unwrap()
        .lines()
        .collect::<Vec<&str>>()
        .chunks(DIGIT_HEIGHT)
        .map(|chunk| chunk.join("\n"))
        .collect();

    let digit_validation_img: Vec<digit::Digit> = digit_vec.into_iter().map(Digit::new).collect();

    let digit_data: Vec<Vec<digit::Digit>> =
        vec![digit_training_img, digit_testing_img, digit_validation_img];

    let digit_training_label: Vec<usize> = fs::read_to_string("./data/digitdata/traininglabels")
        .unwrap()
        .trim()
        .chars()
        .filter_map(|x| {
            if x.is_ascii_digit() {
                return Some(x.to_digit(10).unwrap() as usize);
            }
            None
        })
        .collect();

    let digit_testing_label: Vec<usize> = fs::read_to_string("./data/digitdata/testlabels")
        .unwrap()
        .trim()
        .chars()
        .filter_map(|x| {
            if x.is_ascii_digit() {
                return Some(x.to_digit(10).unwrap() as usize);
            }
            None
        })
        .collect();

    let digit_validation_label: Vec<usize> =
        fs::read_to_string("./data/digitdata/validationlabels")
            .unwrap()
            .trim()
            .chars()
            .filter_map(|x| {
                if x.is_ascii_digit() {
                    return Some(x.to_digit(10).unwrap() as usize);
                }
                None
            })
            .collect();

    let data_labels: Vec<Vec<usize>> = vec![
        digit_training_label,
        digit_testing_label,
        digit_validation_label,
    ];

    let mut digit_perceptron = NeuralNetwork::new(digit_data, data_labels, 1, vec![15]);

    digit_perceptron.start();
}
