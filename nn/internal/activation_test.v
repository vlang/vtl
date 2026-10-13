module internal

import vtl
import math

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

fn test_deriv_sigmoid_contiguous() {
	gradient := vtl.from_1d([2.0, 4.0, 6.0])!
	cached := vtl.from_1d([0.25, 0.5, 0.75])!
	result := deriv_sigmoid[f64](gradient, cached)!

	assert result.array_equal(vtl.from_1d([0.375, 1.0, 1.125])!)
}

fn test_deriv_tanh_contiguous() {
	gradient := vtl.from_1d([2.0, 4.0, 6.0])!
	cached := vtl.from_1d([-0.5, 0.0, 0.5])!
	result := deriv_tanh[f64](gradient, cached)!

	assert result.array_equal(vtl.from_1d([1.5, 4.0, 4.5])!)
}

fn test_deriv_leaky_relu_contiguous_f32() {
	gradient := vtl.from_1d([f32(2.0), 4.0, 6.0])!
	cached := vtl.from_1d([f32(-1.0), 0.0, 2.0])!
	result := deriv_leaky_relu[f32](gradient, cached, f32(0.1))!

	assert result.get_nth(0) == f32(0.2)
	assert result.get_nth(1) == f32(4.0)
	assert result.get_nth(2) == f32(6.0)
}

fn test_deriv_sigmoid_non_contiguous_fallback() {
	gradient := vtl.from_array[f64]([2.0, 4.0, 6.0, 8.0], [2, 2])!.transpose([1, 0])!
	cached := vtl.from_array[f64]([0.25, 0.5, 0.75, 0.0], [2, 2])!.transpose([1, 0])!
	result := deriv_sigmoid[f64](gradient, cached)!

	assert result.get([0, 0]) == 0.375
	assert result.get([0, 1]) == 1.125
	assert result.get([1, 0]) == 1.0
	assert result.get([1, 1]) == 0.0
}

fn test_deriv_gelu_contiguous_f64() {
	gradient := vtl.from_1d([1.0, 2.0, 3.0])!
	cached := vtl.from_1d([-2.0, 0.0, 1.5])!
	result := deriv_gelu[f64](gradient, cached)!
	for i in 0 .. cached.size {
		x := cached.get_nth(i)
		z := 0.7978845608028654 * (x + 0.044715 * x * x * x)
		tanh_z := math.tanh(z)
		dz_dx := 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
		derivative := 0.5 * (1.0 + tanh_z + x * (1.0 - tanh_z * tanh_z) * dz_dx)
		expected := gradient.get_nth(i) * derivative
		assert math.abs(result.get_nth(i) - expected) < 1e-12
	}
}

fn test_deriv_gelu_contiguous_f32() {
	gradient := vtl.from_1d([f32(1), 2, 3])!
	cached := vtl.from_1d([f32(-2), 0, 1.5])!
	result := deriv_gelu[f32](gradient, cached)!
	for i in 0 .. cached.size {
		x := f64(cached.get_nth(i))
		z := 0.7978845608028654 * (x + 0.044715 * x * x * x)
		tanh_z := math.tanh(z)
		dz_dx := 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
		derivative := 0.5 * (1.0 + tanh_z + x * (1.0 - tanh_z * tanh_z) * dz_dx)
		expected := gradient.get_nth(i) * f32(derivative)
		assert math.abs(f64(result.get_nth(i) - expected)) < 1e-6
	}
}

fn test_deriv_gelu_non_contiguous_fallback() {
	gradient := vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!.transpose([1, 0])!
	cached := vtl.from_array([-2.0, 0.0, 1.5, 3.0], [2, 2])!.transpose([1, 0])!
	result := deriv_gelu[f64](gradient, cached)!
	for i in 0 .. cached.size {
		x := cached.get_nth(i)
		z := 0.7978845608028654 * (x + 0.044715 * x * x * x)
		tanh_z := math.tanh(z)
		dz_dx := 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
		derivative := 0.5 * (1.0 + tanh_z + x * (1.0 - tanh_z * tanh_z) * dz_dx)
		expected := gradient.get_nth(i) * derivative
		assert math.abs(result.get_nth(i) - expected) < 1e-12
	}
}

fn test_deriv_gelu_broadcast_fallback() {
	gradient := vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!
	cached := vtl.from_1d([-1.0, 1.0])!
	result := deriv_gelu[f64](gradient, cached)!
	for row in 0 .. 2 {
		for col in 0 .. 2 {
			x := cached.get_nth(col)
			z := 0.7978845608028654 * (x + 0.044715 * x * x * x)
			tanh_z := math.tanh(z)
			dz_dx := 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
			derivative := 0.5 * (1.0 + tanh_z + x * (1.0 - tanh_z * tanh_z) * dz_dx)
			expected := gradient.get([row, col]) * derivative
			assert math.abs(result.get([row, col]) - expected) < 1e-12
		}
	}
}

fn test_deriv_softplus_contiguous_f32_stable_extremes() {
	gradient := vtl.from_1d([f32(1), 2, 3, 4, 5])!
	cached := vtl.from_1d([f32(-100), -1, 0, 1, 100])!
	result := deriv_softplus[f32](gradient, cached)!
	for i in 0 .. cached.size {
		x := f64(cached.get_nth(i))
		sigmoid := if x >= 0 {
			1.0 / (1.0 + math.exp(-x))
		} else {
			exp_x := math.exp(x)
			exp_x / (1.0 + exp_x)
		}
		expected := gradient.get_nth(i) * f32(sigmoid)
		assert math.abs(f64(result.get_nth(i) - expected)) < 1e-6
	}
}

fn test_deriv_softplus_non_contiguous_fallback() {
	gradient := vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!.transpose([1, 0])!
	cached := vtl.from_array([-100.0, -1.0, 1.0, 100.0], [2, 2])!.transpose([1, 0])!
	result := deriv_softplus[f64](gradient, cached)!
	for i in 0 .. cached.size {
		x := cached.get_nth(i)
		sigmoid := if x >= 0 {
			1.0 / (1.0 + math.exp(-x))
		} else {
			exp_x := math.exp(x)
			exp_x / (1.0 + exp_x)
		}
		assert math.abs(result.get_nth(i) - gradient.get_nth(i) * sigmoid) < 1e-12
	}
}
