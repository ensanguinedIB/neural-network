#![allow(dead_code)]

// This `main.rs` is the first file of the network program that Rust will read.

// Each `mod` statement tells Rust to include the relevant file.
// Splitting the code up into files helps keep things organised.

mod dataset;
mod image;
mod network;

// These are things which are imported from the other files in the folder.
// The `use` statement means I don't have to write out the full path later on.

use std::env::args;
use std::fs::File;
use std::io::Read;
use std::process::exit;

use dataset::{IDXError, load_idx1_data, load_idx3_data};
use image::Image;
use network::Network;

// nalgebra is the external library I chose for linear algebra.
// A DVector is a type of vector, i.e. a list.

use nalgebra::DVector;


fn as_output_vec(x: u8) -> DVector<f64>
{
    // Turns a number into a vector the same format as the output of the network

    // The network has 10 outputs, which ideally range between 0 and 1.
    // If the network predicts a 6, it should set the sixth output to 1 and the rest to 0:
    // VALUE:   0   1   2   3   4   5   6   7   8   9
    // OUTPUT:  0   0   0   0   0   0   1   0   0   0

    // This function translates a label, such as 6, to the corresponding list of values.

    // Creates list of 10 0s.
    let mut out = vec![0.0; 10];

    // Sets the xth element to 1.
    out[x as usize] = 1.0;

    // Returns the list as a nalgebra vector for use with the network
    DVector::from(out)
}


fn load_dataset() -> Result<((Vec<DVector<f64>>, Vec<DVector<f64>>), (Vec<DVector<f64>>, Vec<DVector<f64>>)), IDXError>
{
    // This function loads all the dataset files.

    // The load_idx3_data returns a list of Images, so each Image is converted into a nalgebra vector.
    // The load_idx1_data returns labels which are converted using the as_output_vec function above.

    let training_images: Vec<DVector<f64>> = load_idx3_data("./dataset/train-images.idx3-ubyte")?.into_iter().map(|im| DVector::from(im.as_pixel_vec())).collect();
    let training_labels: Vec<DVector<f64>> = load_idx1_data("./dataset/train-labels.idx1-ubyte")?.into_iter().map(|x| as_output_vec(x)).collect();
    let test_images: Vec<DVector<f64>> = load_idx3_data("./dataset/t10k-images.idx3-ubyte")?.into_iter().map(|im| DVector::from(im.as_pixel_vec())).collect();
    let test_labels: Vec<DVector<f64>> = load_idx1_data("./dataset/t10k-labels.idx1-ubyte")?.into_iter().map(|x| as_output_vec(x)).collect();

    // All of the data is returned.
    Ok(((training_images, training_labels), (test_images, test_labels)))
}


fn main()
{
    let path = args().nth(1).unwrap();

    let mut file = File::open(path).unwrap();
    
    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();

    let image = Image::from_vec(data).unwrap();
    let input = DVector::from(image.as_pixel_vec());

    let network = Network::from_file("sig-512-512-0.02.dump").unwrap();
    let output = network.feed_forward(&input);

    exit(output.imax() as i32)
}

// sig-512-0.02:         2,922/10,000
// sig-1024-0.02:        4,728/10,000 
// sig-512-512-0.02:     7,550/10,000
// sig-512-512-0.06:     5,716/10,000
