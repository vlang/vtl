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
	assert repeated_roots == [1.0, 1.0]
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

fn test_chebyshev_evaluation_conversion_calculus_and_fit() ! {
	domain := [f64(-2.0), 2.0]!
	window := [f64(-1.0), 1.0]!
	p := chebyshev([1.0, 2.0, 3.0], domain, window)!
	assert p.degree() == 2
	assert p.eval(-2.0) == 2.0
	assert p.eval(0.0) == -2.0
	assert p.eval(2.0) == 6.0
	assert p.to_polynomial()!.coefficients == [-2.0, 1.0, 1.5]
	round_trip := chebyshev_from_polynomial(p.to_polynomial()!, domain, window)!
	for index, value in p.coefficients {
		assert math.abs(round_trip.coefficients[index] - value) < 1e-12
	}
	input := from_array[f64]([0.0, 2.0], [1, 2])!
	assert p.eval_tensor(input)!.to_array() == [-2.0, 6.0]
	derivative := p.derivative()!
	assert math.abs(derivative.eval(0.0) - 1.0) < 1e-12
	assert math.abs(derivative.eval(1.0) - 4.0) < 1e-12
	integral := p.integral(0.0)!
	for x in [-2.0, -0.5, 1.0, 2.0] {
		assert math.abs(integral.derivative()!.eval(x) - p.eval(x)) < 1e-10
	}
	roots := p.roots()!
	assert roots.len == 2
	for root in roots {
		assert math.abs(p.eval(root)) < 1e-12
	}
	assert p.add(p)!.eval(1.0) == 1.0
	assert p.mul(p)!.eval(0.0) == 4.0

	x := [-2.0, -1.0, 0.0, 1.0, 2.0]
	y := [2.0, -1.5, -2.0, 0.5, 6.0]
	fitted := chebyshev_fit(x, y, 2, domain, window)!
	for index, value in p.coefficients {
		assert math.abs(fitted.coefficients[index] - value) < 1e-10
	}
	if _ := chebyshev([1.0], [1.0, 1.0]!, window) {
		assert false, 'chebyshev must reject zero-width domains'
	}
	if _ := chebyshev([1.0], domain, [0.0, math.inf(1)]!) {
		assert false, 'chebyshev must reject non-finite windows'
	}
	if _ := p.add(chebyshev([1.0], [0.0, 2.0]!, window)!) {
		assert false, 'chebyshev arithmetic must reject mismatched domains'
	}
}

fn test_legendre_evaluation_conversion_calculus_and_fit() ! {
	domain := [f64(-2.0), 2.0]!
	window := [f64(-1.0), 1.0]!
	p := legendre([1.0, 2.0, 3.0], domain, window)!
	assert p.degree() == 2
	assert p.eval(-2.0) == 2.0
	assert p.eval(0.0) == -0.5
	assert p.eval(2.0) == 6.0
	assert p.to_polynomial()!.coefficients == [-0.5, 1.0, 1.125]
	round_trip := legendre_from_polynomial(p.to_polynomial()!, domain, window)!
	for index, value in p.coefficients {
		assert math.abs(round_trip.coefficients[index] - value) < 1e-12
	}
	cubic := legendre([0.5, -1.0, 2.0, 0.75], domain, window)!
	cubic_round_trip := legendre_from_polynomial(cubic.to_polynomial()!, domain, window)!
	for x in [-2.0, -1.2, 0.0, 1.3, 2.0] {
		assert math.abs(cubic_round_trip.eval(x) - cubic.eval(x)) < 1e-12
	}
	input := from_array[f64]([0.0, 2.0], [1, 2])!
	assert p.eval_tensor(input)!.to_array() == [-0.5, 6.0]
	derivative := p.derivative()!
	assert math.abs(derivative.eval(0.0) - 1.0) < 1e-12
	assert math.abs(derivative.eval(1.0) - 3.25) < 1e-12
	integral := p.integral(0.0)!
	for x in [-2.0, -0.5, 1.0, 2.0] {
		assert math.abs(integral.derivative()!.eval(x) - p.eval(x)) < 1e-10
	}
	roots := p.roots()!
	assert roots.len == 2
	for root in roots {
		assert math.abs(p.eval(root)) < 1e-12
	}
	assert p.add(p)!.eval(1.0) == 3.25
	assert math.abs(p.mul(p)!.eval(0.0) - 0.25) < 1e-12

	x := [-2.0, -1.0, 0.0, 1.0, 2.0]
	y := [2.0, -0.375, -0.5, 1.625, 6.0]
	fitted := legendre_fit(x, y, 2, domain, window)!
	for index, value in p.coefficients {
		assert math.abs(fitted.coefficients[index] - value) < 1e-10
	}
	if _ := legendre([1.0], domain, [0.0, 0.0]!) {
		assert false, 'legendre must reject zero-width windows'
	}
	if _ := p.add(legendre([1.0], [0.0, 2.0]!, window)!) {
		assert false, 'legendre arithmetic must reject mismatched domains'
	}
}

fn test_hermite_evaluation_conversion_calculus_and_fit() ! {
	domain := [f64(-2.0), 2.0]!
	window := [f64(-1.0), 1.0]!
	p := hermite([1.0, 2.0, 3.0], domain, window)!
	assert p.degree() == 2
	assert p.eval(-2.0) == 3.0
	assert p.eval(0.0) == -5.0
	assert p.eval(2.0) == 11.0
	assert p.to_polynomial()!.coefficients == [-5.0, 2.0, 3.0]
	round_trip := hermite_from_polynomial(p.to_polynomial()!, domain, window)!
	for index, value in p.coefficients {
		assert math.abs(round_trip.coefficients[index] - value) < 1e-12
	}
	cubic := hermite([0.5, -1.0, 2.0, 0.75], domain, window)!
	cubic_round_trip := hermite_from_polynomial(cubic.to_polynomial()!, domain, window)!
	for x in [-2.0, -1.2, 0.0, 1.3, 2.0] {
		assert math.abs(cubic_round_trip.eval(x) - cubic.eval(x)) < 1e-12
	}
	input := from_array[f64]([0.0, 2.0], [1, 2])!
	assert p.eval_tensor(input)!.to_array() == [-5.0, 11.0]
	derivative := p.derivative()!
	assert math.abs(derivative.eval(0.0) - 2.0) < 1e-12
	assert math.abs(derivative.eval(1.0) - 8.0) < 1e-12
	integral := p.integral(0.0)!
	for x in [-2.0, -0.5, 1.0, 2.0] {
		assert math.abs(integral.derivative()!.eval(x) - p.eval(x)) < 1e-10
	}
	roots := p.roots()!
	assert roots.len == 2
	for root in roots {
		assert math.abs(p.eval(root)) < 1e-12
	}
	assert p.add(p)!.eval(1.0) == 0.0
	assert p.mul(p)!.eval(0.0) == 25.0
	x := [-2.0, -1.0, 0.0, 1.0, 2.0]
	y := [3.0, -4.0, -5.0, 0.0, 11.0]
	fitted := hermite_fit(x, y, 2, domain, window)!
	for index, value in p.coefficients {
		assert math.abs(fitted.coefficients[index] - value) < 1e-10
	}
	if _ := hermite([1.0], domain, [0.0, 0.0]!) {
		assert false, 'hermite must reject zero-width windows'
	}
}

fn test_hermite_e_evaluation_conversion_calculus_and_fit() ! {
	domain := [f64(-2.0), 2.0]!
	window := [f64(-1.0), 1.0]!
	p := hermite_e([1.0, 2.0, 3.0], domain, window)!
	assert p.degree() == 2
	assert p.eval(-2.0) == -1.0
	assert p.eval(0.0) == -2.0
	assert p.eval(2.0) == 3.0
	assert p.to_polynomial()!.coefficients == [-2.0, 1.0, 0.75]
	round_trip := hermite_e_from_polynomial(p.to_polynomial()!, domain, window)!
	for index, value in p.coefficients {
		assert math.abs(round_trip.coefficients[index] - value) < 1e-12
	}
	cubic := hermite_e([0.5, -1.0, 2.0, 0.75], domain, window)!
	cubic_round_trip := hermite_e_from_polynomial(cubic.to_polynomial()!, domain, window)!
	for x in [-2.0, -1.2, 0.0, 1.3, 2.0] {
		assert math.abs(cubic_round_trip.eval(x) - cubic.eval(x)) < 1e-12
	}
	input := from_array[f64]([0.0, 2.0], [1, 2])!
	assert p.eval_tensor(input)!.to_array() == [-2.0, 3.0]
	derivative := p.derivative()!
	assert math.abs(derivative.eval(0.0) - 1.0) < 1e-12
	assert math.abs(derivative.eval(1.0) - 2.5) < 1e-12
	integral := p.integral(0.0)!
	for x in [-2.0, -0.5, 1.0, 2.0] {
		assert math.abs(integral.derivative()!.eval(x) - p.eval(x)) < 1e-10
	}
	roots := p.roots()!
	assert roots.len == 2
	for root in roots {
		assert math.abs(p.eval(root)) < 1e-12
	}
	assert p.add(p)!.eval(1.0) == -0.5
	assert p.mul(p)!.eval(0.0) == 4.0
	x := [-2.0, -1.0, 0.0, 1.0, 2.0]
	y := [-1.0, -2.25, -2.0, -0.25, 3.0]
	fitted := hermite_e_fit(x, y, 2, domain, window)!
	for index, value in p.coefficients {
		assert math.abs(fitted.coefficients[index] - value) < 1e-10
	}
	if _ := hermite_e([1.0], domain, [0.0, 0.0]!) {
		assert false, 'hermite_e must reject zero-width windows'
	}
}

fn test_orthogonal_families_match_numpy_255_reference_values() ! {
	// Reference values come from NumPy 2.5.5's polynomial classes.
	domain := [f64(-2.0), 2.0]!
	window := [f64(-1.0), 1.0]!
	sample_x := [-2.0, -0.5, 0.0, 1.0, 2.0]
	chebyshev_values := [2.0, -2.125, -2.0, 0.5, 6.0]
	chebyshev_power := [-2.0, 1.0, 1.5]
	chebyshev_series := chebyshev([1.0, 2.0, 3.0], domain, window)!
	for index, x in sample_x {
		assert math.abs(chebyshev_series.eval(x) - chebyshev_values[index]) < 1e-12
	}
	for index, coefficient in chebyshev_power {
		assert math.abs(chebyshev_series.to_polynomial()!.coefficients[index] - coefficient) < 1e-12
	}
	chebyshev_derivative := chebyshev_series.derivative()!
	chebyshev_derivative_values := [-5.0, -0.5, 1.0, 4.0, 7.0]
	for index, x in sample_x {
		assert math.abs(chebyshev_derivative.eval(x) - chebyshev_derivative_values[index]) < 1e-12
	}
	assert_roots_match_numpy(chebyshev_series.roots()!, -1.5351837584879966, 0.8685170918213299)
	chebyshev_fit_result := chebyshev_fit(sample_x, chebyshev_values, 2, domain, window)!
	for index, coefficient in [1.0, 2.0, 3.0] {
		assert math.abs(chebyshev_fit_result.coefficients[index] - coefficient) < 1e-12
	}

	legendre_values := [2.0, -0.71875, -0.5, 1.625, 6.0]
	legendre_power := [-0.5, 1.0, 1.125]
	legendre_series := legendre([1.0, 2.0, 3.0], domain, window)!
	for index, x in sample_x {
		assert math.abs(legendre_series.eval(x) - legendre_values[index]) < 1e-12
	}
	for index, coefficient in legendre_power {
		assert math.abs(legendre_series.to_polynomial()!.coefficients[index] - coefficient) < 1e-12
	}
	legendre_derivative := legendre_series.derivative()!
	legendre_derivative_values := [-3.5, -0.125, 1.0, 3.25, 5.5]
	for index, x in sample_x {
		assert math.abs(legendre_derivative.eval(x) - legendre_derivative_values[index]) < 1e-12
	}
	assert_roots_match_numpy(legendre_series.roots()!, -1.24567806121422, 0.3567891723253313)
	legendre_fit_result := legendre_fit(sample_x, legendre_values, 2, domain, window)!
	for index, coefficient in [1.0, 2.0, 3.0] {
		assert math.abs(legendre_fit_result.coefficients[index] - coefficient) < 1e-12
	}

	hermite_values := [3.0, -5.25, -5.0, 0.0, 11.0]
	hermite_power := [-5.0, 2.0, 3.0]
	hermite_series := hermite([1.0, 2.0, 3.0], domain, window)!
	for index, x in sample_x {
		assert math.abs(hermite_series.eval(x) - hermite_values[index]) < 1e-12
	}
	for index, coefficient in hermite_power {
		assert math.abs(hermite_series.to_polynomial()!.coefficients[index] - coefficient) < 1e-12
	}
	hermite_derivative := hermite_series.derivative()!
	hermite_derivative_values := [-10.0, -1.0, 2.0, 8.0, 14.0]
	for index, x in sample_x {
		assert math.abs(hermite_derivative.eval(x) - hermite_derivative_values[index]) < 1e-12
	}
	assert_roots_match_numpy(hermite_series.roots()!, -1.6666666666666667, 1.0)
	hermite_fit_result := hermite_fit(sample_x, hermite_values, 2, domain, window)!
	for index, coefficient in [1.0, 2.0, 3.0] {
		assert math.abs(hermite_fit_result.coefficients[index] - coefficient) < 1e-12
	}

	hermite_e_values := [-1.0, -2.3125, -2.0, -0.25, 3.0]
	hermite_e_power := [-2.0, 1.0, 0.75]
	hermite_e_series := hermite_e([1.0, 2.0, 3.0], domain, window)!
	for index, x in sample_x {
		assert math.abs(hermite_e_series.eval(x) - hermite_e_values[index]) < 1e-12
	}
	for index, coefficient in hermite_e_power {
		assert math.abs(hermite_e_series.to_polynomial()!.coefficients[index] - coefficient) < 1e-12
	}
	hermite_e_derivative := hermite_e_series.derivative()!
	hermite_e_derivative_values := [-2.0, 0.25, 1.0, 2.5, 4.0]
	for index, x in sample_x {
		assert math.abs(hermite_e_derivative.eval(x) - hermite_e_derivative_values[index]) < 1e-12
	}
	assert_roots_match_numpy(hermite_e_series.roots()!, -2.4305008740430605, 1.0971675407097272)
	hermite_e_fit_result := hermite_e_fit(sample_x, hermite_e_values, 2, domain, window)!
	for index, coefficient in [1.0, 2.0, 3.0] {
		assert math.abs(hermite_e_fit_result.coefficients[index] - coefficient) < 1e-12
	}
	repeated_roots := polynomial([1.0, -2.0, 1.0])!.roots()!
	assert math.abs(repeated_roots[0] - 0.9999999999999999) < 1e-12
	assert math.abs(repeated_roots[1] - 0.9999999999999999) < 1e-12
}

fn assert_roots_match_numpy(actual []f64, expected_first f64, expected_second f64) {
	assert actual.len == 2
	in_order := math.abs(actual[0] - expected_first) < 1e-12
		&& math.abs(actual[1] - expected_second) < 1e-12
	reversed := math.abs(actual[0] - expected_second) < 1e-12
		&& math.abs(actual[1] - expected_first) < 1e-12
	assert in_order || reversed
}
