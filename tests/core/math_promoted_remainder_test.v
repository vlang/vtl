module core

import math
import vtl

fn test_promoted_float_remainder_matches_numpy_sign_and_broadcasting() ! {
	values := vtl.from_array[f32]([-5.0, 5.0], [2, 1])!
	divisors := vtl.from_array[f64]([3.0, -3.0], [1, 2])!
	remainders := vtl.remainder_promoted[f64, f32, f64](values, divisors)!

	assert remainders.shape == [2, 2]
	assert remainders.dtype() == .float64
	assert remainders.to_array() == [1.0, -2.0, 2.0, -1.0]
}

fn test_promoted_float_remainder_handles_zero_divisors_and_signed_zero() ! {
	values := vtl.from_1d[f32]([5.0, -5.0, -0.0, 0.0])!
	divisors := vtl.from_1d[f64]([0.0, 0.0, 3.0, -3.0])!
	remainders := vtl.remainder_promoted[f64, f32, f64](values, divisors)!

	assert math.is_nan(remainders.get_nth(0))
	assert math.is_nan(remainders.get_nth(1))
	assert remainders.get_nth(2) == 0.0
	assert !math.signbit(remainders.get_nth(2))
	assert remainders.get_nth(3) == 0.0
	assert math.signbit(remainders.get_nth(3))
}

fn test_promoted_float_remainder_scalar_zero_divisor_returns_nan() ! {
	values := vtl.from_1d[f32]([5.0, -5.0])!
	divisor := vtl.from_1d[f64]([0.0])!
	remainders := vtl.remainder_promoted[f64, f32, f64](values, divisor)!

	assert remainders.shape == [2]
	assert math.is_nan(remainders.get_nth(0))
	assert math.is_nan(remainders.get_nth(1))
}

fn test_promoted_float_remainder_handles_infinities_and_nan() ! {
	infinity := math.inf(1)
	values := vtl.from_1d[f64]([-1.0, 1.0, infinity, 5.0])!
	divisors := vtl.from_1d[f64]([infinity, math.inf(-1), 2.0, math.nan()])!
	remainders := vtl.remainder_promoted[f64, f64, f64](values, divisors)!

	assert remainders.get_nth(0) == infinity
	assert remainders.get_nth(1) == math.inf(-1)
	assert math.is_nan(remainders.get_nth(2))
	assert math.is_nan(remainders.get_nth(3))
}
