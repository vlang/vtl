module vtl

import math

fn test_polynomial_evaluation_arithmetic_and_calculus() ! {
	p := polynomial([1.0, 2.0, 3.0])!
	assert p.eval(2.0) == 17.0
	assert p.add(polynomial([1.0, 1.0])!).coefficients == [2.0, 3.0, 3.0]
	assert p.mul(polynomial([2.0, 1.0])!).coefficients == [2.0, 5.0, 8.0, 3.0]
	assert p.derivative().coefficients == [2.0, 6.0]
	assert p.integral(4.0).coefficients == [4.0, 1.0, 1.0, 1.0]
}

fn test_polynomial_evaluates_tensor_shapes() ! {
	p := polynomial([1.0, 2.0])!
	input := from_array([-1.0, 0.0, 2.0, 3.0], [2, 2])!
	output := p.eval_tensor(input)!
	assert output.shape == [2, 2]
	assert output.to_array() == [-1.0, 1.0, 5.0, 7.0]
}

fn test_polynomial_roots_and_fit() ! {
	quadratic := polynomial([2.0, -3.0, 1.0])!
	roots := quadratic.roots()!
	assert math.abs(roots[0] - 1.0) < 1e-12
	assert math.abs(roots[1] - 2.0) < 1e-12
	repeated_roots := polynomial([1.0, -2.0, 1.0])!.roots()!
	assert repeated_roots == [1.0]
	ill_scaled_roots := polynomial([1e-16, 1e8, 1.0])!.roots()!
	assert math.abs(ill_scaled_roots[0] + 1e8) < 1e-8
	assert math.abs(ill_scaled_roots[1] + 1e-24) < 1e-32
	fit := polynomial_fit([-1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 2.0, 5.0], 2)!
	assert math.abs(fit.coefficients[0] - 1.0) < 1e-12
	assert math.abs(fit.coefficients[1]) < 1e-12
	assert math.abs(fit.coefficients[2] - 1.0) < 1e-12
}

fn test_polynomial_fit_scales_large_inputs_and_rejects_invalid_data() ! {
	x := [10000.0, 10001.0, 10002.0, 10003.0, 10004.0]
	y := [9.0, 2.0, 1.0, 6.0, 17.0]
	fit := polynomial_fit(x, y, 2)!
	for i, value in x {
		assert math.abs(fit.eval(value) - y[i]) < 1e-4
	}
	if _ := polynomial_fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], 1) {
		assert false, 'polynomial_fit must reject rank-deficient samples'
	}
	if _ := polynomial_fit([0.0, math.nan()], [1.0, 2.0], 1) {
		assert false, 'polynomial_fit must reject non-finite x values'
	}
	if _ := polynomial_fit([0.0, 1.0], [1.0, math.inf(1)], 1) {
		assert false, 'polynomial_fit must reject non-finite y values'
	}
}
