module main

import vtl

fn main() {
	labels := vtl.from_1d([2, 0, 1, 2])!
	encoded := vtl.one_hot(labels, 3)!
	println('labels: ${labels.to_array()}')
	println('one-hot shape: ${encoded.shape}')
	println('one-hot values: ${encoded.to_array()}')
}
