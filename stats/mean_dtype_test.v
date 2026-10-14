module stats

import math
import vtl

fn test_mean_accumulator_dtype_global() {
	values := vtl.from_1d([f32(16_777_216), 1, -16_777_216])!
	assert mean_as[f32, f32](values) == 0.0
	assert math.abs(mean_as[f32, f64](values) - 1.0 / 3.0) < 1e-12
	assert mean_as[int, f64](vtl.from_1d([1, 2, 3])!) == 2.0
	assert math.is_nan(mean_as[f32, f64](vtl.from_1d([]f32{})!))
}

fn test_mean_accumulator_dtype_axis_and_multiple_axes() ! {
	values := vtl.from_array([f32(1), 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	axis := mean_along_axis_as[f32, f64](values, -1, true)!
	assert axis.shape == [2, 2, 1]
	assert axis.to_array() == [1.5, 3.5, 5.5, 7.5]

	multiple := mean_along_axes_as[f32, f64](values, [0, 2], false)!
	assert multiple.shape == [2]
	assert multiple.to_array() == [3.5, 5.5]
	all_axes := mean_along_axes_as[f32, f64](values, [0, 1, 2], false)!
	assert all_axes.get_nth(0) == 4.5
	all_axes_kept := mean_along_axes_as[f32, f64](values, [0, 1, 2], true)!
	assert all_axes_kept.shape == [1, 1, 1]
	assert all_axes_kept.get_nth(0) == 4.5

	unchanged := mean_along_axes_as[f32, f64](values, [], false)!
	assert unchanged.shape == values.shape
	assert unchanged.to_array() == [1, 2, 3, 4, 5, 6, 7, 8]

	matrix := vtl.from_array([f32(1), 2, 3, 4], [2, 2])!
	transposed := matrix.t()!
	transposed_mean := mean_along_axis_as[f32, f64](transposed, 0, false)!
	assert transposed_mean.to_array() == [1.5, 3.5]

	if _ := mean_along_axes_as[f32, f64](values, [0, -3], false) {
		assert false, 'duplicate axes must return an error'
	}
}

fn test_mean_accumulator_dtype_empty_reductions() ! {
	empty := vtl.zeros[f32]([2, 0, 3], vtl.TensorData{})
	means := mean_along_axis_as[f32, f64](empty, 1, false)!
	assert means.shape == [2, 3]
	for value in means.to_array() {
		assert math.is_nan(value)
	}

	kept := mean_along_axes_as[f32, f64](empty, [1], true)!
	assert kept.shape == [2, 1, 3]
	for value in kept.to_array() {
		assert math.is_nan(value)
	}
}
