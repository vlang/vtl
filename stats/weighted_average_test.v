module stats

import vtl

fn test_average_uses_weights_and_supports_integer_tensors() ! {
	values := vtl.from_1d([1, 2, 3])!
	weights := vtl.from_1d([1.0, 1.0, 2.0])!
	assert average(values, weights)! == 2.25
}

fn test_average_axis_accepts_axis_weights_and_full_weights() ! {
	values := vtl.from_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!
	axis_weights := vtl.from_1d([1.0, 1.0, 2.0])!
	rows := average_axis(values, axis_weights, 1)!
	assert rows.shape == [2, 1]
	assert rows.to_array() == [2.25, 5.25]
	full_weights := vtl.from_2d([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0]])!
	columns := average_axis(values, full_weights, 0)!
	assert columns.shape == [1, 3]
	assert columns.to_array() == [3.0, 3.0, 4.5]
}

fn test_average_along_axis_selects_keepdims_shape() ! {
	values := vtl.from_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!
	weights := vtl.from_1d([1.0, 1.0, 2.0])!
	rows := average_along_axis(values, weights, -1, false)!
	assert rows.shape == [2]
	assert rows.to_array() == [2.25, 5.25]
	columns := average_along_axis(values, vtl.ones[f64]([2, 3]), 0, true)!
	assert columns.shape == [1, 3]
	assert columns.to_array() == [2.5, 3.5, 4.5]
	vector := vtl.from_1d([2.0, 4.0])!
	scalar := average_along_axis(vector, vtl.ones[f64]([2]), 0, false)!
	assert scalar.rank() == 0
	assert scalar.get_nth(0) == 3.0
}

fn test_average_along_multiple_axes_with_full_weights() ! {
	values := vtl.from_array[f64]([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	weights := vtl.from_array[f64]([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	means := average_along_axes(values, weights, [0, -1], false)!
	assert means.shape == [2]
	assert means.to_array() == [66.0 / 14.0, 138.0 / 22.0]
	kept := average_along_axes(values, weights, [0, 2], true)!
	assert kept.shape == [1, 2, 1]
	assert kept.to_array() == means.to_array()
	scalar := average_along_axes(values, weights, [0, 1, 2], false)!
	assert scalar.rank() == 0
	assert scalar.get_nth(0) == 204.0 / 36.0
}

fn test_average_along_multiple_axes_rejects_invalid_inputs() ! {
	values := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	weights := vtl.ones[f64]([2, 2])
	if _ := average_along_axes(values, weights, [0, -2], false) {
		assert false, 'duplicate axes must fail'
	}
	if _ := average_along_axes(values, weights, [2], false) {
		assert false, 'out-of-range axes must fail'
	}
	if _ := average_along_axes(values, vtl.ones[f64]([2]), [0], false) {
		assert false, 'weights with a mismatched shape must fail'
	}
	if _ := average_along_axes(values, vtl.from_2d([[1.0, -1.0], [1.0, -1.0]])!, [1], false) {
		assert false, 'zero-sum slice weights must fail'
	}
	empty := vtl.from_array[f64]([]f64{}, [2, 0])!
	if _ := average_along_axes(empty, empty, [1], false) {
		assert false, 'empty averaged axes must fail'
	}
}

fn test_average_rejects_empty_mismatched_and_zero_weight_inputs() ! {
	values := vtl.from_1d([1.0, 2.0])!
	wrong_shape := vtl.from_2d([[1.0, 2.0]])!
	if _ := average(values, wrong_shape) {
		assert false, 'different shapes must return an error'
	} else {
		assert true
	}
	zero_weights := vtl.from_1d([1.0, -1.0])!
	if _ := average(values, zero_weights) {
		assert false, 'zero-sum weights must return an error'
	} else {
		assert true
	}
	if _ := average_axis(values, wrong_shape, 0) {
		assert false, 'invalid axis weight shape must return an error'
	} else {
		assert true
	}
}
