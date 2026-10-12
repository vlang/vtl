module internal

import vtl

fn test_deriv_relu_contiguous_f32() {
	gradient := vtl.from_1d([f32(1.0), 2.0, 3.0, 4.0])!
	cached := vtl.from_1d([f32(-1.0), 0.0, 2.0, -3.0])!
	result := deriv_relu[f32](gradient, cached)!

	assert result.array_equal(vtl.from_1d([f32(0.0), 2.0, 3.0, 0.0])!)
}

fn test_deriv_relu_contiguous_f64_empty() {
	gradient := vtl.from_1d([]f64{})!
	cached := vtl.from_1d([]f64{})!
	result := deriv_relu[f64](gradient, cached)!

	assert result.shape == gradient.shape
	assert result.size == 0
}

fn test_deriv_relu_non_contiguous_fallback() {
	gradient := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [2, 2])!.transpose([1, 0])!
	cached := vtl.from_array[f64]([-1.0, 0.0, 2.0, -3.0], [2, 2])!.transpose([1, 0])!
	result := deriv_relu[f64](gradient, cached)!

	assert result.get([0, 0]) == 0.0
	assert result.get([0, 1]) == 3.0
	assert result.get([1, 0]) == 2.0
	assert result.get([1, 1]) == 0.0
}

fn test_deriv_relu_broadcast_fallback() {
	gradient := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [2, 2])!
	cached := vtl.from_1d([-1.0, 2.0])!
	result := deriv_relu[f64](gradient, cached)!

	assert result.get([0, 0]) == 0.0
	assert result.get([0, 1]) == 2.0
	assert result.get([1, 0]) == 0.0
	assert result.get([1, 1]) == 4.0
}
