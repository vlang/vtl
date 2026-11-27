module core

import vtl
import math.complex

fn test_promoted_boolean_binary_arithmetic_matches_numpy() ! {
	flags := vtl.from_1d[bool]([true, false])!
	other := vtl.from_1d[bool]([true, true])!

	assert vtl.add_promoted[bool, bool, bool](flags, other)!.to_array() == [true, true]
	assert vtl.multiply_promoted[bool, bool, bool](flags, other)!.to_array() == [
		true,
		false,
	]
	if _ := vtl.subtract_promoted[bool, bool, bool](flags, other) {
		assert false, 'boolean subtraction must be rejected'
	}

	integers := vtl.from_1d[i8]([3, 4])!
	assert vtl.add_promoted[i8, bool, i8](flags, integers)!.to_array() == [4, 4]
	assert vtl.subtract_promoted[i8, bool, i8](flags, integers)!.to_array() == [-2, -4]
	assert vtl.multiply_promoted[i8, bool, i8](flags, integers)!.to_array() == [3, 0]

	assert vtl.divide_promoted[f64, bool, bool](flags, other)!.to_array() == [1.0, 0.0]
	assert vtl.floor_divide_promoted[i8, bool, bool](flags, other)!.to_array() == [
		i8(1),
		0,
	]
	assert vtl.fmod_promoted[i8, bool, bool](flags, other)!.to_array() == [i8(0), 0]
	assert vtl.remainder_promoted[i8, bool, bool](flags, other)!.to_array() == [i8(0), 0]
	assert vtl.pow_promoted[i8, bool, bool](flags, other)!.to_array() == [i8(1), 0]
}

fn test_binary_promoted_auto_selects_runtime_result_dtype() ! {
	flags := vtl.from_1d[bool]([true, false])!
	integers := vtl.from_1d[i8]([3, 4])!

	sum := vtl.binary_promoted_auto(flags, integers, .add)!
	assert vtl.promoted_tensor_dtype(sum) == .int8
	assert sum is &vtl.Tensor[i8]
	assert (sum as &vtl.Tensor[i8]).to_array() == [i8(4), 4]

	boolean_product := vtl.binary_promoted_auto(flags, flags, .multiply)!
	assert vtl.promoted_tensor_dtype(boolean_product) == .boolean
	assert boolean_product is &vtl.Tensor[bool]
	assert (boolean_product as &vtl.Tensor[bool]).to_array() == [true, false]

	denominators := vtl.from_1d[bool]([true, true])!
	quotients := vtl.binary_promoted_auto(flags, denominators, .divide)!
	assert vtl.promoted_tensor_dtype(quotients) == .float64
	assert quotients is &vtl.Tensor[f64]
	assert (quotients as &vtl.Tensor[f64]).to_array() == [1.0, 0.0]

	powers := vtl.binary_promoted_auto(flags, flags, .power)!
	assert vtl.promoted_tensor_dtype(powers) == .int8
	assert (powers as &vtl.Tensor[i8]).to_array() == [i8(1), 1]

	complex_values := vtl.from_1d[complex.Complex]([complex.complex(1.0, 2.0)])!
	complex_sum := vtl.binary_promoted_auto(complex_values, flags, .add)!
	assert vtl.promoted_tensor_dtype(complex_sum) == .complex128
	assert (complex_sum as &vtl.Tensor[complex.Complex]).to_array() == [
		complex.complex(2.0, 2.0),
		complex.complex(1.0, 2.0),
	]
}

fn test_promoted_boolean_weak_scalars_use_numpy_dtypes() ! {
	flags := vtl.from_1d[bool]([true, false])!

	added := vtl.add_scalar_promoted[bool, bool, bool](flags, true)!
	assert added.to_array() == [true, true]
	auto_added := vtl.scalar_promoted_auto(flags, true, .add)!
	assert vtl.promoted_tensor_dtype(auto_added) == .boolean
	assert (auto_added as &vtl.Tensor[bool]).to_array() == [true, true]

	multiplied := vtl.multiply_scalar_promoted[int, bool, int](flags, 2)!
	assert multiplied.dtype() == .native_int
	assert multiplied.to_array() == [2, 0]
	auto_multiplied := vtl.scalar_promoted_auto(flags, 2, .multiply)!
	assert vtl.promoted_tensor_dtype(auto_multiplied) == .native_int
	assert (auto_multiplied as &vtl.Tensor[int]).to_array() == [2, 0]

	subtracted := vtl.subtract_scalar_left_promoted[int, bool, int](1, flags)!
	assert subtracted.dtype() == .native_int
	assert subtracted.to_array() == [0, 1]
	auto_subtracted := vtl.scalar_left_promoted_auto(1, flags, .subtract)!
	assert vtl.promoted_tensor_dtype(auto_subtracted) == .native_int
	assert (auto_subtracted as &vtl.Tensor[int]).to_array() == [0, 1]

	powered := vtl.pow_scalar_promoted[i8, bool, bool](flags, true)!
	assert powered.dtype() == .int8
	assert powered.to_array() == [i8(1), 0]
	auto_powered := vtl.scalar_promoted_auto(flags, true, .power)!
	assert vtl.promoted_tensor_dtype(auto_powered) == .int8
	assert (auto_powered as &vtl.Tensor[i8]).to_array() == [i8(1), 0]

	float_powered := vtl.scalar_promoted_auto(flags, 2, .float_power)!
	assert vtl.promoted_tensor_dtype(float_powered) == .float64
	assert (float_powered as &vtl.Tensor[f64]).to_array() == [1.0, 0.0]

	complex_result := vtl.scalar_promoted_auto(flags, complex.complex(0.0, 1.0), .add)!
	assert vtl.promoted_tensor_dtype(complex_result) == .complex128
	assert (complex_result as &vtl.Tensor[complex.Complex]).to_array() == [
		complex.complex(1.0, 1.0),
		complex.complex(0.0, 1.0),
	]
}
