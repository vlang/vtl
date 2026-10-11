module core

import math
import vtl

fn test_masked_arg_extrema_return_first_unmasked_flat_index() ! {
	values := vtl.from_1d([8.0, 100, 2, 8])!
	mask := vtl.from_1d[bool]([false, true, false, false])!
	data := vtl.masked_array(values, mask)!

	maximum := data.argmax()
	minimum := data.argmin()
	assert maximum.value == 0 && !maximum.is_masked
	assert minimum.value == 2 && !minimum.is_masked

	all_masked := vtl.masked_array(values, vtl.ones[bool]([4]))!
	assert all_masked.argmax().value == 0 && all_masked.argmax().is_masked
	assert all_masked.argmin().value == 0 && all_masked.argmin().is_masked
	empty := vtl.masked_array(vtl.zeros[f64]([0]), vtl.zeros[bool]([0]))!
	assert empty.argmax().is_masked
	assert empty.argmin().is_masked

	integers := vtl.masked_array(vtl.from_1d([4, 6, 1])!, vtl.zeros[bool]([3]))!
	assert integers.argmax().value == 1
	assert integers.argmin().value == 2
}

fn test_masked_arg_extrema_reduce_axis_keepdims_and_mark_empty_slices() ! {
	values := vtl.from_array[f64]([1, 99, 4, 7, 8, 6], [2, 3])!
	mask := vtl.from_array[bool]([false, true, false, true, true, true], [2, 3])!
	data := vtl.masked_array(values, mask)!

	maxima := data.argmax_along_axis(-1, true)!
	minima := data.argmin_along_axis(1, false)!
	assert maxima.values.shape == [2, 1]
	assert maxima.values.to_array() == [2, 0]
	assert maxima.mask.to_array() == [false, true]
	assert minima.values.shape == [2]
	assert minima.values.to_array() == [0, 0]
	assert minima.mask.to_array() == [false, true]
}

fn test_masked_arg_extrema_follow_nan_first_index_and_validate_axes() ! {
	values := vtl.from_array[f64]([1, math.nan(), 3, math.nan(), 2, 4], [2, 3])!
	mask := vtl.from_array[bool]([false, false, false, false, true, false], [2, 3])!
	data := vtl.masked_array(values, mask)!
	assert data.argmax().value == 1
	assert data.argmin().value == 1
	assert data.argmax_along_axis(1, false)!.values.to_array() == [1, 0]
	assert data.argmin_along_axis(1, false)!.values.to_array() == [1, 0]
	if _ := data.argmax_along_axis(2, false) {
		assert false, 'masked arg reductions must reject an out-of-bounds axis'
	}
}

fn test_masked_arg_extrema_use_logical_indices_for_strided_views() ! {
	values := vtl.from_2d([[1, 4], [7, 2]])!.transpose([1, 0])!
	mask := vtl.zeros[bool]([2, 2]).transpose([1, 0])!
	data := vtl.masked_array(values, mask)!

	assert data.argmax().value == 1
	assert data.argmin().value == 0
	assert data.argmax_along_axis(1, false)!.values.to_array() == [1, 0]
	assert data.argmin_along_axis(1, false)!.values.to_array() == [0, 1]
}
