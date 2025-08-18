use crate::generic_lib::{HasPixels, PixelSizes, DIGIT_HEIGHT, DIGIT_LENGTH};

#[derive(Clone)]
pub struct Digit {
    pixels: [[char; DIGIT_LENGTH]; DIGIT_HEIGHT],
    label: Option<usize>,
}

impl Digit {
    pub fn new(input: String) -> Self {
        let mut pixels = [[' '; DIGIT_LENGTH]; DIGIT_HEIGHT];
        let input: Vec<char> = input.chars().collect();
        // for y in 0..DIGIT_HEIGHT {
        //     for x in 0..DIGIT_LENGTH {
        //         let index = y * DIGIT_LENGTH + x;
        //         pixels[y][x] = input[index];
        //     }
        // }
        //
        for (y, row) in pixels.iter_mut().enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                let index = y * DIGIT_LENGTH + x;
                *pixel = input[index];
            }
        }

        Digit {
            pixels,
            label: None,
        }
    }
}
impl HasPixels for Digit {
    fn get_pixels(&self) -> PixelSizes {
        PixelSizes::PixelWrapper(self.pixels)
    }

    fn get_size(&self) -> usize {
        DIGIT_LENGTH * DIGIT_HEIGHT
    }

    fn get_num_output(&self) -> usize {
        10
    }

    fn flatten(&self) -> Vec<f64> {
        let mut result: Vec<f64> = Vec::new();
        for x in self.pixels {
            for y in x {
                if y.is_whitespace() {
                    result.push(0.0);
                } else {
                    result.push(1.0);
                }
            }
        }

        result
    }
}
