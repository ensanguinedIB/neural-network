#![allow(dead_code)]

use std::fs::File;
use std::fmt::{Display, Formatter};
use std::io::Read;

use crate::image::{Image, ImageError};


pub type Result<T> = std::result::Result<T, IDXError>;


#[derive(Debug)]
#[repr(u32)]
enum MagicNumber
{
    IDX1 = 2049,
    IDX3 = 2051
}


pub fn load_idx1_data(path: &str) -> Result<Vec<u8>>
{
    let mut data = Vec::new();
    let mut file = File::open(path)?;

    file.read_to_end(&mut data)?;

    let first_four_bytes: [u8; 4] = data[0..4].try_into()?;
    let magic_number = u32::from_be_bytes(first_four_bytes);

    if magic_number != MagicNumber::IDX1 as u32
    {
        return Err(IDXError::MagicNumberError);
    }

    let second_four_bytes: [u8; 4] = data[4..8].try_into()?;
    let size = u32::from_be_bytes(second_four_bytes);

    let labels = data.split_off(8);
    let length = labels.len();

    if length != size.try_into()?
    {
        return Err(IDXError::LengthError);
    }

    Ok(labels)
}

pub fn load_idx3_data(path: &str) -> Result<Vec<Image>>
{
    let mut data = Vec::new();
    let mut file = File::open(path)?;

    file.read_to_end(&mut data)?;

    let first_four_bytes: [u8; 4] = data[0..4].try_into()?;
    let magic_number = u32::from_be_bytes(first_four_bytes);

    if magic_number != MagicNumber::IDX3 as u32
    {
        return Err(IDXError::MagicNumberError);
    }

    let number_of_images = u32::from_be_bytes((&data[4..8]).try_into()?);
    let row_width = u32::from_be_bytes((&data[8..12]).try_into()?);
    let column_height = u32::from_be_bytes((&data[12..16]).try_into()?);

    let image_size = row_width * column_height;

    let image_data = data.split_off(16);
    let mut chunks = image_data.chunks(image_size.try_into()?);

    let mut images = Vec::new();

    while let Some(pixels) = chunks.next()
    {
        let image = Image::from_vec(pixels.to_vec())?;
        images.push(image);
    }

    let length = images.len();

    if length != number_of_images.try_into()?
    {
        return Err(IDXError::LengthError);
    }

    Ok(images)
}


// This is a custom error type which is returned from the load_idxN_data methods

#[derive(Debug)]
pub enum IDXError
{
    MagicNumberError,
    LengthError,
    IOError(std::io::Error),
    TryFromIntError(std::num::TryFromIntError),
    TryFromSliceError(std::array::TryFromSliceError),
    ImageError(ImageError)
}


// Everything below is boilerplate which must be implemented for an error type.
// Display handles printing the error, and the From blocks allow for quiet conversions between error types.

impl Display for IDXError
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Error reading IDX data: ")?;

        match self
        {
            IDXError::MagicNumberError => write!(f, "incorrect magic number for this file format."),
            IDXError::LengthError => write!(f, "number of items did not match specified file length."),
            IDXError::IOError(e) => write!(f, "{}.", e),
            IDXError::TryFromIntError(e) => write!(f, "{}.", e),
            IDXError::TryFromSliceError(e) => write!(f, "{}.", e),
            IDXError::ImageError(e) => write!(f, "{}", e)
        }
    }
}

impl From<std::io::Error> for IDXError
{
    fn from(e: std::io::Error) -> IDXError
    {
        IDXError::IOError(e)
    }
}

impl From<std::num::TryFromIntError> for IDXError
{
    fn from(e: std::num::TryFromIntError) -> IDXError
    {
        IDXError::TryFromIntError(e)
    }
}

impl From<std::array::TryFromSliceError> for IDXError
{
    fn from(e: std::array::TryFromSliceError) -> IDXError
    {
        IDXError::TryFromSliceError(e)
    }
}

impl From<ImageError> for IDXError
{
    fn from(e: ImageError) -> IDXError
    {
        IDXError::ImageError(e)
    }
}
