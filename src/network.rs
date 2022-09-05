#![allow(dead_code)]

// This is the main file of the program, which defines the Network.
// The Network can 'learn', predict, and save and load itself to and from files.

// Imports:
//
// File, Read, and Write are used to read/write files to the disc.
// They are in the standard (std) library, which means Rust provides them.
//
// Rng is a random number generator, from the `rand` crate, and thread_rng() creates the generator.
// The `rand` crate is the go-to crate for randomness, as the Rust standard library does not provide this.
//
// `nalgebra` is the crate I chose to do the operations on vectors and matrices.
// I didn't have any particular reasoning for choosing this library.

use std::fs::File;
use std::io::{Read, Write};

use rand::{Rng, thread_rng};
use nalgebra::{DMatrix, DVector};

// A struct is a holder for related information.
// They usually provide methods, which operate on the struct's properties.
// There is more on the neural network theory in the progress commentary.

#[derive(Debug)]
pub struct Network
{
    pub learning_rate: f64,             // The learning rate determines how large changes to the weights and biases are.
    weights: Vec<DMatrix<f64>>,     // The weights and biases operate on the inputs to generate the output.
    biases: Vec<DVector<f64>>,
    activation: ActivationFunction  // The activation function limits the output in some way (e.g. by 'squishing' it between 0 and 1).
}


// The `impl` block defines functionality which create a Network struct or modify an existing one in some way.
// There are two creation functions: `new` and `from_file`, and three other methods:
// `dump` saves the Network to a file, `feed_forward` is for predicting once the network is trained, and `train` is for training the network.

impl Network
{
    pub fn new(layer_sizes: &[usize], activation_function: ActivationFunction, learning_rate: Option<f64>) -> Network
    {
        // The `new` function is used to create a Network struct from scratch.
        // Most of the function is concerned with randomly generating the weights and biases.
        //
        // The parts inside the brackets are the parameters, which the user must provide when creating a Network:
        // `layer_sizes` is a list ẁhich details how many neurons should be in each layer.  For example, [5, 1] is a network with five inputs and one output.
        // `activation_function` tells the network which activation function to use, and can be Identity, ReLU, or Sigmoid.  See the ActivationFunction enum.
        // `learning_rate` is a number, normally less than zero, which modifies how fast the network should learn.  Small learning rates mean smaller adjustments.

        // The `learning_rate` parameter is an Option, which means the user may not provide it.
        // The Network should always have a learning rate, so if the user doesn't give one, default to 0.02.
        let learning_rate = learning_rate.unwrap_or(0.02);

        // Create a new random number generator called `rng`.
        let mut rng = thread_rng();

        // Create two new empty lists.  We will add randomised matrices to the `weights` list, and randomised vectors to the `bias` list.
        // The lists are ordered so that the first element in `weights` and the first element in `biases` correspond to the first layer.
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // The `.windows(2)` method slides along the list showing 2 elements at a time.
        // If the layer sizes were defined by the list [5, 6, 4, 1] then the windows would be [5, 6], [6, 4], and [4, 1].
        // Each layer has a weight matrix, and matrices are two dimensional - the matrix for the first layer should be 6x5 in our example.
        let mut dimensions = layer_sizes.windows(2);

        // This while loop says for each pair of dimensions, do the code in the block.
        while let Some(&[dim0, dim1]) = dimensions.next()
        {
            // These next two lines construct a random matrix and a vector.  The first arguments are the required dimensions.
            // The last argument in each case is a function which always returns a random number between -1 and 1 inclusive.
            let weight_matrix = DMatrix::from_fn(dim1, dim0, |_, _| rng.gen_range(-1.0..=1.0));
            let bias_vector = DVector::from_fn(dim1, |_, _| rng.gen_range(-1.0..=1.0));

            // The matrix and the vector are added to the end of their respective lists.
            weights.push(weight_matrix);
            biases.push(bias_vector);
        }
        
        // At the end of the loop, the correct number of weight matrices and bias vectors should have been created (one less than the number of layers).

        // Constructing the Network involves filling in all of the fields with the variables we just created.
        // This Network struct is returned from the function.
        Network
        {
            learning_rate: learning_rate,
            weights: weights,
            biases: biases,
            activation: activation_function
        }
    }

    pub fn from_file(path: &str) -> Result<Network, std::io::Error>
    {
        // This function creates a new network based on saved data in a file.
        // It is effectively the opposite of the dump method.

        // First the file is opened and an empty list is initialised.
        let mut file = File::open(path)?;
        let mut data = Vec::new();

        // The data is read from the file as bytes
        file.read_to_end(&mut data)?;

        // The first eight bytes contain the learning rate, this is converted into a decimal value
        let learning_rate_bytes = data[0..8].try_into().unwrap();
        let learning_rate = f64::from_be_bytes(learning_rate_bytes);

        // The activation function is defined by the next byte
        let activation_byte = data[8];

        // Match the byte to one of the given activation functions
        let activation = match activation_byte
        {
            0 => ActivationFunction::Identity,
            1 => ActivationFunction::ReLU,
            2 => ActivationFunction::Sigmoid,
            _ => panic!("encountered unexpected value decoding network dump file")
        };

        // Initialise empty lists for storing weights and biases in
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Ignore the first nine bytes as they have already been processed
        let mut data = data.split_off(9);

        // Read until there are no more bytes in data
        while !data.is_empty()
        {
            // The data is encoded as "layer 1 weight matrix" "layer 1 bias vector" "layer 2 weight matrix" "layer 2 bias vector" ...

            // The width and height of the vector are encoded in the first eight bytes
            let width_bytes: Vec<u8> = data.drain(0..4).collect();
            let width_bytes: [u8; 4] = width_bytes.try_into().unwrap();
            let width = u32::from_be_bytes(width_bytes);

            let height_bytes: Vec<u8> = data.drain(0..4).collect();
            let height_bytes: [u8; 4] = height_bytes.try_into().unwrap();
            let height = u32::from_be_bytes(height_bytes);

            // The number of elements in the matrix is the width × the height
            let length = (width * height * 8) as usize;
            
            // Read the appropriate number of bytes from the data (each element is eight bytes) and turn them into decimals
            let weight_bytes: Vec<u8> = data.drain(0..length).collect();
            let weight_values: Vec<f64> = weight_bytes.chunks(8).map(|x| f64::from_be_bytes(x.try_into().unwrap())).collect();

            // Create a new matrix from the values and add it to the list
            let matrix = DMatrix::from_vec(width as usize, height as usize, weight_values);
            weights.push(matrix);


            // Vectors only have a length, so four bytes declare the number of elements to read.
            let length_bytes: Vec<u8> = data.drain(0..4).collect();
            let length_bytes: [u8; 4] = length_bytes.try_into().unwrap();
            let length = (u32::from_be_bytes(length_bytes) * 8) as usize;

            // Read the appropriate number of bytes (again, 8 bytes per value) then convert into numbers
            let bias_bytes: Vec<u8> = data.drain(0..length).collect();
            let bias_bytes: Vec<f64> = bias_bytes.chunks(8).map(|x| f64::from_be_bytes(x.try_into().unwrap())).collect();

            // Create a vector from the values and add that to its list
            let vector = DVector::from_vec(bias_bytes);
            biases.push(vector);
        }

        // Create a new network and return it

        let network = Network
        {
            learning_rate: learning_rate,
            weights: weights,
            biases: biases,
            activation: activation,
        };

        Ok(network)
    }

    pub fn dump(&self, path: &str) -> Result<(), std::io::Error>
    {
        // The `dump` method saves the Network to a file.
        // It stores the learning rate, activation function, and the weights and biases.
        // I came up with the simple file format myself, and explain it in the code commentary.
        //
        // The method takes one parameter, `path`.
        // This is the path to the file that the user wants to save to.  It will be overwritten if it already exists.

        // Try to create a file with at the given path.
        let mut file = File::create(path)?;
        
        // Create an empty list.  This will hold the bytes to be written to the file.
        let mut data: Vec<u8> = Vec::new();

        // Each of the data stored in the network, with the exception of the activation function, span multiple bytes.
        // Instead of interpreting the numbers as their values, Rust can provide a list of the bytes using `.to_be_bytes()`.

        // The `.extend()` and `.push()` methods are both adding bytes to the end of the `data` list.
        // The learning rate is 8 bytes long, as are the other floating-point (decimal) numbers.
        data.extend(self.learning_rate.to_be_bytes());
        // There are only three activation functions, so ActivationFunction fits inside a single byte and can cast directly.
        data.push(self.activation as u8);

        // Get the pairs of weights and biases from each layer.  For each pair, do the code in the block.
        for (weights, biases) in self.weights.iter().zip(self.biases.iter())
        {
            // `.shape()` gets the dimensions of the weight matrix.
            let (weights_width, weights_height) = weights.shape();

            // Add the width and height of the matrix to the data.  Each is 4 bytes long.
            data.extend((weights_width as u32).to_be_bytes());
            data.extend((weights_height as u32).to_be_bytes());

            // Flatten the matrix (a 6x8 matrix becomes a list of 48 things) and add each value to the data.
            data.extend(weights.iter().map(|x| x.to_be_bytes()).flatten());

            // The same deal for the bias vector.  First find its length
            let bias_length = biases.len();
            
            // Write the length to the data.
            data.extend((bias_length as u32).to_be_bytes());

            // Write the bias vector to the data.  A vector is effectively already a list.
            data.extend(biases.iter().map(|x| x.to_be_bytes()).flatten());
        }

        // Write all of the data to the file and finish.
        file.write(&data)?;

        Ok(())
    }

    pub fn feed_forward(&self, inputs: &DVector<f64>) -> DVector<f64>
    {
        // This method makes a prediction using the current weights and biases

        // We need to keep track of the activations in each layer to pass forward to the next one.
        let mut activation = inputs.to_owned();

        // Get the weights and biases of each layer in turn
        for (weights, biases) in self.weights.iter().zip(self.biases.iter())
        {
            // These are the steps mentioned in the "Understanding machine learning" section.
            // The weighted sum is calculated then passed through the activation function.
            let weighted_sum = weights * activation + biases;
            activation = (self.activation.forward())(weighted_sum);
        }

        activation
    }

    pub fn train(&mut self, inputs: &[DVector<f64>], outputs: &[DVector<f64>])
    {
        // It is expected that the training data is passed as a long list, so iterate over each image and label in the list
        for (input, expected) in inputs.iter().zip(outputs.iter())
        {
            // This begins in much the same way as the feed forward algorithm
            // The weighted sums (what goes in the activation function) has to be stored too, though
            let mut activations = vec![input.to_owned()];
            let mut weighted_sums = Vec::new();

            for (weights, biases) in self.weights.iter().zip(self.biases.iter())
            {
                let weighted_sum = weights * activations.last().unwrap() + biases;
                let activation = (self.activation.forward())(weighted_sum.clone());

                weighted_sums.push(weighted_sum);
                activations.push(activation);
            }

            let actual = activations.last().unwrap();
            
            // The actual error of the network doesn't need to be calculated, but the correct gradient (slope) of the error function does
            let d_cost_wrt_output = 2.0 * (actual - expected);

            // This keeps track of the slopes at each layer in the network.
            let mut local_gradients = vec![d_cost_wrt_output];

            // Go through each layer backwards
            for i in (0..self.weights.len()).rev()
            {
                // These variables are what is fed into the gradient equations
                let previous_layer_output = activations.get(i).unwrap();
                let this_layer_sum = weighted_sums.get(i).unwrap();
                let this_layer_weights = self.weights.get_mut(i).unwrap();
                let this_layer_biases = self.biases.get_mut(i).unwrap();
                
                // This calculates the local gradient for this layer.
                // The local gradient is initially the slope of the error function, but gets more complex as the layers go back.
                let local_gradient = local_gradients.last().unwrap().clone().component_mul(&(self.activation.backward())(this_layer_sum.clone()));

                // Calculate the gradient of the weights and biases in this layer
                let d_cost_wrt_weights = &local_gradient * previous_layer_output.transpose();
                let d_cost_wrt_biases = &local_gradient;

                let d_cost_wrt_previous_layer = this_layer_weights.transpose() * &local_gradient;
                local_gradients.push(d_cost_wrt_previous_layer);

                // Adjust the weights and biases of the network slightly
                *this_layer_weights -= self.learning_rate * d_cost_wrt_weights;
                *this_layer_biases -= self.learning_rate * d_cost_wrt_biases;
            }
        }
    }
}


// This is an enum (can only take certain values) which declares which activation functions are available.
// The sigmoid is explained in the progress commentary, and the identity function simply does nothing to the value.
// The ReLU is another common function but I didn't use it

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum ActivationFunction
{
    Identity = 0,
    ReLU = 1,
    Sigmoid = 2
}

// The activation functions need to be used, so these methods define both the functions and the derivatives (slope functions) of the functions.

impl ActivationFunction
{
    pub fn forward(self) -> fn(DVector<f64>) -> DVector<f64>
    {
        match self
        {
            // The part after the arrow is the function.
            // Identity maps the value to itself so v -> v
            ActivationFunction::Identity => |v: DVector<f64>| v,
            ActivationFunction::ReLU => |v: DVector<f64>| v.map(|x| x.max(0.0)),
            ActivationFunction::Sigmoid => |v: DVector<f64>| v.map(|x| 1.0 / (1.0 + (-x).exp()))
        }
    }

    pub fn backward(self) -> fn(DVector<f64>) -> DVector<f64>
    {
        match self
        {
            ActivationFunction::Identity => |v: DVector<f64>| v.map(|_| 1.0),
            ActivationFunction::ReLU => |v: DVector<f64>| v.map(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::Sigmoid => |v: DVector<f64>| v.map(|x| (-x).exp() / (1.0 + (-x).exp()).powi(2))
        }
    }
}
