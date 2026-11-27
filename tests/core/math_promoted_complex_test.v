module core

import math.complex
import vtl

fn test_promoted_complex_add_broadcasts_real_tensor() ! {
	complex_values := vtl.from_array[complex.Complex]([
		complex.complex(1.0, 2.0),
		complex.complex(-1.0, 1.0),
	], [2, 1])!
	real_values := vtl.from_1d[f32]([1.0, 2.0, -3.0])!
	result := vtl.add_promoted[complex.Complex, complex.Complex, f32](complex_values,
		real_values)!

	assert result.dtype() == .complex128
	assert result.shape == [2, 3]
	expected := [
		complex.complex(2.0, 2.0),
		complex.complex(3.0, 2.0),
		complex.complex(-2.0, 2.0),
		complex.complex(0.0, 1.0),
		complex.complex(1.0, 1.0),
		complex.complex(-4.0, 1.0),
	]
	for index, value in expected {
		actual := result.get_nth(index)
		assert actual.re == value.re
		assert actual.im == value.im
	}
}

fn test_promoted_complex_arithmetic_supports_mixed_inputs() ! {
	complex_values := vtl.from_1d[complex.Complex]([
		complex.complex(1.0, 2.0),
		complex.complex(3.0, -1.0),
	])!
	complex_offsets := vtl.from_1d[complex.Complex]([
		complex.complex(0.5, -0.5),
		complex.complex(-1.0, 2.0),
	])!
	real_values := vtl.from_1d[f64]([2.0, 4.0])!

	sum := vtl.add_promoted[complex.Complex, complex.Complex, complex.Complex](complex_values,
		complex_offsets)!
	assert sum.get_nth(0).re == 1.5
	assert sum.get_nth(0).im == 1.5
	assert sum.get_nth(1).re == 2.0
	assert sum.get_nth(1).im == 1.0

	difference := vtl.subtract_promoted[complex.Complex, complex.Complex, f64](complex_values,
		real_values)!
	assert difference.get_nth(0).re == -1.0
	assert difference.get_nth(0).im == 2.0
	assert difference.get_nth(1).re == -1.0
	assert difference.get_nth(1).im == -1.0

	product := vtl.multiply_promoted[complex.Complex, f64, complex.Complex](real_values,
		complex_values)!
	assert product.get_nth(0).re == 2.0
	assert product.get_nth(0).im == 4.0
	assert product.get_nth(1).re == 12.0
	assert product.get_nth(1).im == -4.0

	quotient := vtl.divide_promoted[complex.Complex, complex.Complex, f64](complex_values,
		real_values)!
	assert quotient.get_nth(0).re == 0.5
	assert quotient.get_nth(0).im == 1.0
	assert quotient.get_nth(1).re == 0.75
	assert quotient.get_nth(1).im == -0.25
}

fn test_promoted_complex_arithmetic_reads_strided_views() ! {
	base := vtl.from_array[complex.Complex]([
		complex.complex(1.0, 1.0),
		complex.complex(2.0, 2.0),
		complex.complex(3.0, 3.0),
		complex.complex(4.0, 4.0),
	], [2, 2])!
	transposed := base.transpose([1, 0])!
	offset := vtl.from_1d[i8]([10, -10])!
	result := vtl.add_promoted[complex.Complex, complex.Complex, i8](transposed, offset)!

	assert result.shape == [2, 2]
	assert result.get_nth(0).re == 11.0
	assert result.get_nth(0).im == 1.0
	assert result.get_nth(1).re == -7.0
	assert result.get_nth(1).im == 3.0
	assert result.get_nth(2).re == 12.0
	assert result.get_nth(2).im == 2.0
	assert result.get_nth(3).re == -6.0
	assert result.get_nth(3).im == 4.0
}

fn test_promoted_complex_arithmetic_validates_output_dtype_and_rejects_remainder() ! {
	real_values := vtl.from_1d[f32]([1.0])!
	complex_values := vtl.from_1d[complex.Complex]([complex.complex(1.0, 2.0)])!

	if _ := vtl.add_promoted[complex.Complex, f32, f32](real_values, real_values) {
		assert false, 'complex output must not be accepted for real-only operands'
	} else {
		assert err.msg().contains('output dtype')
	}

	if _ := vtl.remainder_promoted[complex.Complex, complex.Complex, f32](complex_values,
		real_values) {
		assert false, 'complex remainder must be rejected'
	} else {
		assert err.msg().contains('remainder')
	}
}
