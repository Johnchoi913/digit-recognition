pub const DIGIT_LENGTH: usize = 28;
pub const DIGIT_HEIGHT: usize = 28;

pub trait HasPixels {
    fn get_pixels(&self) -> PixelSizes;
    fn get_size(&self) -> usize;
    fn get_num_output(&self) -> usize;
    fn flatten(&self) -> Vec<f64>;
}

pub enum PixelSizes {
    PixelWrapper([[char; DIGIT_LENGTH]; DIGIT_HEIGHT]),
}

// pub trait Classifer {
//     fn dot(&self, v) -> f64 {

//     }
// }
