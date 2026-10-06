module core

import math
import vtl

fn test_array_equal_float_values_are_exact() ! {
	exact := vtl.from_1d([f64(1.0), 2.0])!
	same := vtl.from_1d([f64(1.0), 2.0])!
	nearby := vtl.from_1d([f64(1.0 + 1e-12), 2.0])!

	assert exact.array_equal(same)
	assert !exact.array_equal(nearby)
}

fn test_array_equal_uses_ieee_nan_infinity_and_zero_behavior() ! {
	nan := vtl.from_1d([math.nan()])!
	positive_infinity := vtl.from_1d([math.inf(1)])!
	matching_infinity := vtl.from_1d([math.inf(1)])!
	positive_zero := vtl.from_1d([0.0])!
	negative_zero := vtl.from_1d([-0.0])!

	assert !nan.array_equal(nan)
	assert positive_infinity.array_equal(matching_infinity)
	assert positive_zero.array_equal(negative_zero)
}

fn test_array_equal_requires_matching_shapes() ! {
	vector := vtl.from_1d([f64(1.0), 2.0])!
	matrix := vtl.from_2d([[f64(1.0), 2.0]])!

	assert !vector.array_equal(matrix)
}
