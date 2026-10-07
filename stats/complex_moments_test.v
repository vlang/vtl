module stats

import math
import math.complex
import vtl

fn test_complex_axis_moments_keepdims_and_dtype() ! {
	values := vtl.from_array[complex.Complex]([
		complex.Complex{ re: 1, im: 2 },
		complex.Complex{ re: 3, im: 4 },
		complex.Complex{ re: 5, im: 6 },
		complex.Complex{ re: 7, im: 8 },
	], [2, 2])!
	means := complex_mean_along_axis(values, 0, false)!
	assert means.shape == [2]
	assert means.dtype() == .complex128
	assert means.to_array() == [complex.Complex{ re: 3, im: 4 }, complex.Complex{ re: 5, im: 6 }]
	variances := complex_variance_along_axis(values, 1, 0, true)!
	assert variances.shape == [2, 1]
	assert variances.to_array() == [2.0, 2.0]
	deviations := complex_std_along_axis(values, 0, 1, false)!
	assert deviations.shape == [2]
	assert deviations.to_array() == [4.0, 4.0]
}

fn test_complex_multi_axis_moments_and_transposed_layout() ! {
	values := vtl.from_array[complex.Complex]([
		complex.Complex{ re: 1, im: 0 },
		complex.Complex{ re: 2, im: 0 },
		complex.Complex{ re: 3, im: 0 },
		complex.Complex{ re: 4, im: 0 },
		complex.Complex{ re: 5, im: 0 },
		complex.Complex{ re: 6, im: 0 },
		complex.Complex{ re: 7, im: 0 },
		complex.Complex{ re: 8, im: 0 },
	], [2, 2, 2])!
	means := complex_mean_along_axes(values, [0, 2], true)!
	assert means.shape == [1, 2, 1]
	assert means.to_array() == [complex.Complex{ re: 3.5, im: 0 }, complex.Complex{ re: 5.5, im: 0 }]
	variances := complex_variance_along_axes(values, [0, -1], 0, false)!
	assert variances.shape == [2]
	assert variances.to_array() == [4.25, 4.25]
	transposed := values.transpose([2, 1, 0])!
	transposed_means := complex_mean_along_axes(transposed, [0, 2], false)!
	assert transposed_means.to_array() == means.reshape[complex.Complex]([2])!.to_array()
}

fn test_complex_moments_empty_axes_empty_slices_and_errors() ! {
	values := vtl.from_array[complex.Complex]([complex.Complex{ re: 2, im: 3 },
		complex.Complex{ re: -1, im: 4 }], [2])!
	unchanged := complex_mean_along_axes(values, [], false)!
	assert unchanged.shape == values.shape
	assert unchanged.to_array() == values.to_array()
	zero_variance := complex_variance_along_axes(values, [], 0, false)!
	assert zero_variance.to_array() == [0.0, 0.0]
	empty := vtl.from_array[complex.Complex]([]complex.Complex{}, [2, 0])!
	empty_means := complex_mean_along_axis(empty, 1, false)!
	assert empty_means.shape == [2]
	assert math.is_nan(empty_means.get_nth(0).re)
	empty_variances := complex_variance_along_axis(empty, 1, 0, false)!
	assert math.is_nan(empty_variances.get_nth(0))
	if _ := complex_mean_along_axes(values, [0, -1], false) {
		assert false, 'duplicate axes must fail'
	}
	if _ := complex_variance_along_axis(values, 1, -1, false) {
		assert false, 'negative ddof must fail'
	}
}
