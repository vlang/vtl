module nn

import vnum.num
import vnum.la

type ActivateFn = fn (a &num.NdArray) num.NdArray

pub enum Activator {
	sigmoid
	htan
	relu
	softplus
}

struct Weights {
mut:
	input_hidden  num.NdArray
	output_hidden num.NdArray
}

struct Results {
mut:
	hidden_sum    num.NdArray
	hidden_result num.NdArray
	output_sum    num.NdArray
	output_result num.NdArray
}

struct Network {
	learning_rate  f64
	iterations     int
	hidden_units   int
mut:
	activate       ActivateFn
	activate_prime ActivateFn
	weights        Weights
	results        Results
}

pub fn new(rate f64, iterations, units int, activator Activator) Network {
	mut m := Network{
		learning_rate: rate
		iterations: iterations
		hidden_units: units
	}
	match activator {
		.sigmoid {
			m.activate = sigmoid_map
			m.activate_prime = sigmoid_prime_map
		}
		.htan {
			m.activate = htan_map
			m.activate_prime = htan_prime_map
		}
		.relu {
			m.activate = relu_map
			m.activate_prime = relu_prime_map
		}
		.softplus {
			m.activate = softplus_map
			m.activate_prime = softplus_prime_map
		}
	}
	return m
}

pub fn (mut n Network) learn(inputs, outputs num.NdArray) {
	incols := inputs.shape[1]
	outcols := outputs.shape[1]
	n.weights.input_hidden = normals(incols, n.hidden_units)
	n.weights.output_hidden = normals(n.hidden_units, outcols)
	for i := 0; i < n.iterations; i++ {
		n.forward(inputs)
		n.back(inputs, outputs)
	}
}

pub fn (mut n Network) forward(input num.NdArray) {
	hidden_sum := la.matmul(input, n.weights.input_hidden)
	n.results.hidden_result = n.activate(&hidden_sum)
	output_sum := la.matmul(n.results.hidden_result, n.weights.output_hidden)
	n.results.output_result = n.activate(&output_sum)
	n.results.hidden_sum = hidden_sum
	n.results.output_sum = output_sum
}

pub fn (mut n Network) back(input, output num.NdArray) {
	error_output_layer := num.subtract(output, n.results.output_result)
	delta_output_layer := num.multiply(n.activate_prime(&n.results.output_sum), error_output_layer)
	mut hidden_output_changes := la.matmul(n.results.hidden_result.t(), delta_output_layer)
	hidden_output_changes = num.multiply_as(hidden_output_changes, n.learning_rate)
	n.weights.output_hidden = num.add(hidden_output_changes, n.weights.output_hidden)
	mut delta_hidden_layer := la.matmul(delta_output_layer, n.weights.output_hidden.t())
	delta_hidden_layer = num.multiply(n.activate_prime(&n.results.hidden_sum), delta_hidden_layer)
	mut input_hidden_changes := la.matmul(input.t(), delta_hidden_layer)
	input_hidden_changes = num.multiply_as(input_hidden_changes, n.learning_rate)
	n.weights.input_hidden = num.add(input_hidden_changes, n.weights.input_hidden)
}

pub fn (mut n Network) predict(input num.NdArray) num.NdArray {
	n.forward(input)
	return n.results.output_result
}
