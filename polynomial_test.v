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
	fit := polynomial_fit([-1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 2.0, 5.0], 2)!
	assert fit.coefficients == [1.0, 0.0, 1.0]
}
