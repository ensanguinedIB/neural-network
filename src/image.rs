#![allow(dead_code)]

// Allows me to use Display and Formatter without qualifying (std::fmt::Display).
use std::fmt::{Display, Formatter};

// Specify that the default Result type for this file should use the ImageError type.
pub type Result<T> = std::result::Result<T, ImageError>;


// This is a very basic object, just used to hold the pixel data of the images.

#[derive(Debug)]
pub struct Image
{
    bytes: Vec<u8>
}


#[derive(Debug)]
pub struct ImageError
{
    message: String
}


impl Image
{
    pub fn from_vec(bytes: Vec<u8>) -> Result<Image>
    {
        // Create a new Image, given the pixels in a list.

        // All of the images should be 784 pixels long, so complain if not.
        let length = bytes.len();

        if length != 784
        {
            let message = format!("expected 784 bytes, got {}", length);
            return Err(ImageError::new(message));
        }

        // Create the image and return it

        let image = Image
        {
            bytes: bytes
        };

        Ok(image)
    }

    pub fn as_pixel_vec(self) -> Vec<f64>
    {
        // This method turns the image back into a list of pixels.
        // Each byte is turned into a decimal by dividing by 255 so the network can use it.
        
        self.bytes.into_iter().map(|x| x as f64 / 255.0).collect()
    }
}


impl ImageError
{
    fn new(msg: String) -> ImageError
    {
        ImageError { message: msg }
    }
}

impl Display for ImageError
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Error creating Image: {}.", self.message)
    }
}
