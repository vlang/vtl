module stats

import math
import vtl

fn test_variance_and_std_explicit_dtype_global() ! {
	values := vtl.from_1d([f32(10_000), 10_001, 10_002])!
	variance_f32 := variance_as[f32, f32](values, 0)!
	variance_f64 := variance_as[f32, f64](values, 0)!
	assert math.abs(f64(variance_f64) - 2.0 / 3.0) < 1e-12
	assert math.abs(f64(variance_f32) - 2.0 / 3.0) < 0.01
	assert math.abs(f64(std_as[f32, f64](values, 0)!) - math.sqrt(2.0 / 3.0)) < 1e-12
	assert math.abs(variance_as[int, f64](vtl.from_1d([1, 2, 3])!, 1)! - 1.0) < 1e-12
	if _ := variance_as[f32, f64](vtl.from_1d([]f32{})!, 0) {
		assert false, 'variance of an empty tensor must fail'
	}
}

fn test_variance_and_std_explicit_dtype_axis_reductions() ! {
	values := vtl.from_array([f32(1), 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	axis := variance_along_axis_as[f32, f64](values, -1, 1, true)!
	assert axis.shape == [2, 2, 1]
	assert axis.to_array() == [0.5, 0.5, 0.5, 0.5]
	stds := std_along_axis_as[f32, f64](values, 2, 0, false)!
	assert stds.shape == [2, 2]
	assert stds.to_array() == [0.5, 0.5, 0.5, 0.5]

	multi := variance_along_axes_as[f32, f64](values, [0, 2], 0, false)!
	assert multi.shape == [2]
	assert multi.to_array() == [4.25, 4.25]
	assert math.abs(std_along_axes_as[f32, f64](values, [0, 1, 2], 0, false)!.get_nth(0) -
		math.sqrt(5.25)) < 1e-12

	unchanged := variance_along_axes_as[f32, f64](values, [], 0, false)!
	assert unchanged.shape == values.shape
	assert unchanged.to_array() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

	if _ := variance_along_axes_as[f32, f64](values, [0, -3], 0, false) {
		assert false, 'duplicate axes must return an error'
	}
}

fn test_variance_explicit_dtype_empty_and_insufficient_slices() ! {
	empty := vtl.zeros[f32]([2, 0, 3], vtl.TensorData{})
	variances := variance_along_axis_as[f32, f64](empty, 1, 0, false)!
	assert variances.shape == [2, 3]
	for value in variances.to_array() {
		assert math.is_nan(value)
	}

	values := vtl.from_array([f32(1), 3, 5, 9], [2, 2])!
	insufficient := variance_along_axis_as[f32, f64](values, 1, 2, false)!
	for value in insufficient.to_array() {
		assert math.is_nan(value)
	}
}

fn test_nan_moments_explicit_dtype_global() ! {
	values := vtl.from_1d([f32(1), f32(math.nan()), 3, 5])!
	assert nanmean_as[f32, f64](values) == 3.0
	assert math.abs(nanvariance_as[f32, f64](values, 0)! - 8.0 / 3.0) < 1e-12
	assert math.abs(nanstd_as[f32, f64](values, 0)! - math.sqrt(8.0 / 3.0)) < 1e-12

	all_nan := vtl.from_1d([f32(math.nan()), f32(math.nan())])!
	assert math.is_nan(nanmean_as[f32, f64](all_nan))
	assert math.is_nan(nanvariance_as[f32, f64](all_nan, 0)!)
	if _ := nanvariance_as[f32, f64](values, -1) {
		assert false, 'negative ddof must return an error'
	}
}

fn test_nan_moments_explicit_dtype_axis_reductions() ! {
	values := vtl.from_array([f32(1), f32(math.nan()), f32(math.nan()), f32(math.nan())], [
		2,
		2,
	])!
	means := nanmean_along_axis_as[f32, f64](values, 1, false)!
	assert means.shape == [2]
	assert means.get_nth(0) == 1.0
	assert math.is_nan(means.get_nth(1))

	variances := nanvariance_along_axis_as[f32, f64](values, 1, 1, true)!
	assert variances.shape == [2, 1]
	assert math.is_nan(variances.get_nth(0))
	assert math.is_nan(variances.get_nth(1))

	multi_mean := nanmean_along_axes_as[f32, f64](values, [0, 1], false)!
	assert multi_mean.get_nth(0) == 1.0
	multi_std := nanstd_along_axes_as[f32, f64](values, [0, 1], 0, true)!
	assert multi_std.shape == [1, 1]
	assert multi_std.get_nth(0) == 0.0
}
