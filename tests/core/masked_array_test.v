module core

import math
import vtl

fn test_masked_array_broadcast_fill_compress_and_global_reductions() ! {
	values := vtl.from_array[f64]([1, 2, 3, 4, 5, 6], [2, 3])!
	mask := vtl.from_array[bool]([false, true, false, true, true, true], [2, 3])!
	data := vtl.masked_array(values, mask)!
	assert data.values.shape == [2, 3]
	assert data.count() == 2
	assert data.filled(-1.0).to_array() == [1, -1, 3, -1, -1, -1]
	assert data.compressed()!.to_array() == [1, 3]
	sum := data.sum()
	assert sum.value == 4.0 && !sum.is_masked
	mean := data.mean()
	assert mean.value == 2.0 && !mean.is_masked

	row_mask := vtl.from_array[bool]([true, false, true], [3])!
	broadcast_data := vtl.masked_array(values, row_mask)!
	assert broadcast_data.mask.shape == [2, 3]
	assert broadcast_data.compressed()!.to_array() == [2, 5]
	assert broadcast_data.filled(0.0).to_array() == [0, 2, 0, 0, 5, 0]
}

fn test_masked_array_axis_reductions_preserve_masked_empty_slices() ! {
	values := vtl.from_array[int]([1, 2, 3, 4, 5, 6], [2, 3])!
	mask := vtl.from_array[bool]([false, true, false, true, true, true], [2, 3])!
	data := vtl.masked_array(values, mask)!
	sums := data.sum_along_axis(0, false)!
	assert sums.values.shape == [3]
	assert sums.values.to_array() == [1, 0, 3]
	assert sums.mask.to_array() == [false, true, false]
	means := data.mean_along_axis(-1, true)!
	assert means.values.shape == [2, 1]
	assert means.values.to_array()[0] == 2.0
	assert math.is_nan(means.values.to_array()[1])
	assert means.mask.to_array() == [false, true]
}

fn test_masked_array_all_masked_and_invalid_shapes_or_axes() ! {
	values := vtl.from_1d([2.0, 4.0])!
	all_masked := vtl.masked_array(values, vtl.from_1d[bool]([true, true])!)!
	sum := all_masked.sum()
	assert sum.value == 0 && sum.is_masked
	mean := all_masked.mean()
	assert math.is_nan(mean.value) && mean.is_masked
	assert all_masked.count() == 0
	assert all_masked.compressed()!.size == 0
	if _ := vtl.masked_array(values, vtl.from_1d[bool]([true, false, true])!) {
		assert false, 'masked_array must reject incompatible shapes'
	}
	if _ := all_masked.mean_along_axis(1, false) {
		assert false, 'masked axis reduction must reject out-of-range axes'
	}
}

fn test_masked_array_broadcasts_empty_dimensions() ! {
	values := vtl.zeros[f64]([0], vtl.TensorData{})
	mask := vtl.from_1d[bool]([false])!
	data := vtl.masked_array(values, mask)!
	assert data.values.shape == [0]
	assert data.mask.shape == [0]
	assert data.count() == 0
	assert data.compressed()!.size == 0
}
