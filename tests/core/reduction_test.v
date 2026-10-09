module core

import vtl
import math

fn test_argmax_axis_0() {
	t := vtl.from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.argmax_axis(0)!
	// along axis 0: col0 max at row1 (3>1), col1 max at row0 (5>2)
	assert result.get_nth(0) == 1
	assert result.get_nth(1) == 0
}

fn test_argmax_axis_1() {
	t := vtl.from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.argmax_axis(1)!
	// along axis 1: row0 max at col1 (5>1), row1 max at col0 (3>2)
	assert result.get_nth(0) == 1
	assert result.get_nth(1) == 0
}

fn test_argmin_axis_0() {
	t := vtl.from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.argmin_axis(0)!
	// along axis 0: col0 min at row0 (1<3), col1 min at row1 (2<5)
	assert result.get_nth(0) == 0
	assert result.get_nth(1) == 1
}

fn test_argmin_axis_1() {
	t := vtl.from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.argmin_axis(1)!
	// along axis 1: row0 min at col0 (1<5), row1 min at col1 (2<3)
	assert result.get_nth(0) == 0
	assert result.get_nth(1) == 1
}

fn test_arg_extrema_axis_squeeze_matches_numpy_shape() {
	t := vtl.from_2d([[1.0, 5.0], [3.0, 2.0]])!
	max_indices := t.argmax_axis_squeeze(1)!
	assert max_indices.shape == [2]
	assert max_indices.to_array() == [1, 0]
	min_indices := t.argmin_axis_squeeze(-2)!
	assert min_indices.shape == [2]
	assert min_indices.to_array() == [0, 1]

	vector := vtl.from_1d([4.0, 2.0, 6.0])!
	assert vector.argmax_axis_squeeze(0)!.shape == [1]
	assert vector.argmin_axis_squeeze(0)!.to_array() == [1]
}

fn test_argmax_flat() {
	t := vtl.from_1d([1.0, 7.0, 3.0, 5.0])!
	result := t.argmax(0)!
	assert result.get_nth(0) == 1
}

fn test_argmin_flat() {
	t := vtl.from_1d([4.0, 2.0, 6.0, 1.0])!
	result := t.argmin(0)!
	assert result.get_nth(0) == 3
}

fn test_nanarg_flattened_skips_nan_and_keeps_first_tie() {
	t := vtl.from_1d[f64]([math.nan(), 7.0, 2.0, 7.0])!
	assert t.nanargmax()! == 1
	assert t.nanargmin()! == 2
}

fn test_nanarg_flattened_errors_when_no_valid_values() {
	all_nan := vtl.from_1d[f64]([math.nan(), math.nan()])!
	if _ := all_nan.nanargmax() {
		assert false, 'nanargmax must reject all-NaN input'
	}
	if _ := all_nan.nanargmin() {
		assert false, 'nanargmin must reject all-NaN input'
	}
	empty := vtl.from_1d[f64]([]f64{})!
	if _ := empty.nanargmax() {
		assert false, 'nanargmax must reject empty input'
	}
}

fn test_nanarg_axis_skips_nan_and_selects_keepdims_shape() {
	t := vtl.from_2d[f64]([[
		math.nan(),
		8.0,
		3.0,
	], [
		4.0,
		math.nan(),
		1.0,
	]])!
	max_rows := t.nanargmax_axis(1, false)!
	assert max_rows.shape == [2]
	assert max_rows.to_array() == [1, 0]
	min_columns := t.nanargmin_axis(0, true)!
	assert min_columns.shape == [1, 3]
	assert min_columns.to_array() == [1, 0, 1]
}

fn test_nanarg_axis_errors_for_all_nan_slices_and_empty_axes() {
	all_nan_slice := vtl.from_2d[f64]([[math.nan(), math.nan()], [1.0, 2.0]])!
	if _ := all_nan_slice.nanargmax_axis(1, false) {
		assert false, 'axis reduction must reject an all-NaN slice'
	}
	empty_axis := vtl.zeros[f64]([2, 0])
	if _ := empty_axis.nanargmin_axis(1, false) {
		assert false, 'axis reduction must reject an empty axis'
	}
}

fn test_nanarg_axis_handles_strided_views() {
	base := vtl.from_2d[f64]([[math.nan(), 4.0, 2.0], [7.0, math.nan(), 1.0]])!
	transposed := base.transpose([1, 0])!
	max_rows := transposed.nanargmax_axis(1, false)!
	assert max_rows.to_array() == [1, 0, 0]
	min_columns := transposed.nanargmin_axis(0, true)!
	assert min_columns.shape == [1, 2]
	assert min_columns.to_array() == [2, 2]
}

fn test_max_axis_1() {
	t := vtl.from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.max_axis(1)!
	assert result.get_nth(0) == f64(5)
	assert result.get_nth(1) == f64(3)
}

fn test_min_axis_1() {
	t := vtl.from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.min_axis(1)!
	assert result.get_nth(0) == f64(1)
	assert result.get_nth(1) == f64(2)
}

fn test_min_max_axis_squeeze_matches_numpy_shape() {
	t := vtl.from_2d([[1.0, 5.0], [3.0, 2.0]])!
	maxima := t.max_axis_squeeze(1)!
	assert maxima.shape == [2]
	assert maxima.to_array() == [5.0, 3.0]
	minima := t.min_axis_squeeze(-2)!
	assert minima.shape == [2]
	assert minima.to_array() == [1.0, 2.0]

	vector := vtl.from_1d([4.0, 2.0, 6.0])!
	assert vector.max_axis_squeeze(0)!.to_array() == [6.0]
}

fn test_min_max_multi_axis_reductions_match_numpy_shapes() {
	t := vtl.from_array([0.0, 5.0, 3.0, 7.0, 4.0, 2.0, 8.0, 1.0], [2, 2, 2])!
	maxima := t.max_axes([0, -1], false)!
	assert maxima.shape == [2]
	assert maxima.to_array() == [5.0, 8.0]
	minima := t.min_axes([0, 2], true)!
	assert minima.shape == [1, 2, 1]
	assert minima.to_array() == [0.0, 1.0]
	unchanged := t.min_axes([], false)!
	assert unchanged.shape == t.shape
	assert unchanged.to_array() == t.to_array()
	if _ := t.max_axes([0, -3], false) {
		assert false, 'duplicate axes must return an error'
	}
	empty_axis := vtl.from_array([]f64{}, [2, 0])!
	if _ := empty_axis.max_axes([1], false) {
		assert false, 'extrema over empty axes must return an error'
	}
	empty_output := vtl.from_array([]f64{}, [0, 2])!.max_axes([1], false)!
	assert empty_output.shape == [0]
}

fn test_cumsum() {
	t := vtl.from_1d([1.0, 2.0, 3.0, 4.0])!
	result := t.cumsum(0)!
	assert result.get_nth(0) == f64(1)
	assert result.get_nth(1) == f64(3)
	assert result.get_nth(2) == f64(6)
	assert result.get_nth(3) == f64(10)
}

fn test_cumprod() {
	t := vtl.from_1d([1.0, 2.0, 3.0, 4.0])!
	result := t.cumprod(0)!
	assert result.get_nth(0) == f64(1)
	assert result.get_nth(1) == f64(2)
	assert result.get_nth(2) == f64(6)
	assert result.get_nth(3) == f64(24)
}
