module main

import math.complex
import vtl

fn main() {
	bases := vtl.from_2d[i16]([[2], [3]]) or { panic(err) }
	exponents := vtl.from_1d[f32]([2.0, 3.0, 0.0]) or { panic(err) }
	result := vtl.pow_promoted[f32, i16, f32](bases, exponents) or { panic(err) }

	println('shape: ${result.shape}')
	println('dtype: ${result.dtype()}')
	println('powers: ${result.to_array()}')

	auto_result := vtl.binary_promoted_auto(bases, exponents, .power) or { panic(err) }
	println('automatic dtype: ${vtl.promoted_tensor_dtype(auto_result)}')
	if auto_result is &vtl.Tensor[f32] {
		auto_powers := auto_result as &vtl.Tensor[f32]
		println('automatic powers: ${auto_powers.to_array()}')
	} else {
		panic('power promotion should produce float32')
	}

	scalar_result := vtl.scalar_left_promoted_auto(10, bases, .subtract) or { panic(err) }
	println('weak-scalar dtype: ${vtl.promoted_tensor_dtype(scalar_result)}')
	if scalar_result is &vtl.Tensor[i16] {
		scalar_difference := scalar_result as &vtl.Tensor[i16]
		println('weak-scalar difference: ${scalar_difference.to_array()}')
	} else {
		panic('in-range integer scalar should preserve int16')
	}

	complex_bases := vtl.from_1d[complex.Complex]([
		complex.complex(-1.0, 0.0),
		complex.complex(-1.0, -0.0),
	]) or { panic(err) }
	complex_exponents := vtl.from_1d[complex.Complex]([
		complex.complex(0.5, 0.0),
		complex.complex(0.5, 0.0),
	]) or { panic(err) }
	complex_roots := vtl.pow_promoted[complex.Complex, complex.Complex, complex.Complex](
		complex_bases, complex_exponents,
	) or { panic(err) }
	println('complex square roots: ${complex_roots.to_array()}')
}
