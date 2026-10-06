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

fn test_array_equal_does_not_use_absolute_float_epsilon() ! {
	f64_negative := vtl.from_1d([f64(-1.602176634e-19)])!
	f64_positive := vtl.from_1d([f64(9.1093837015e-31)])!
	f32_negative := vtl.from_1d([f32(-1.602176634e-19)])!
	f32_positive := vtl.from_1d([f32(9.1093837015e-31)])!

	assert !f64_negative.array_equal(f64_positive)
	assert !f32_negative.array_equal(f32_positive)
	assert f64_negative.get_nth(0) < f64_positive.get_nth(0)
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
