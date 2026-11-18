module main

import vtl

fn main() {
	values := vtl.from_1d([0.0, 2.0, 10.0])!
	tensor_right := vtl.logaddexp_scalar(values, 1.0)!
	scalar_left := vtl.logaddexp_scalar_left(1.0, values)!

	println('Tensor + scalar: ${tensor_right.to_array()}')
	println('Scalar + tensor: ${scalar_left.to_array()}')
}
