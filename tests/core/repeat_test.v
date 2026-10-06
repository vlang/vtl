module core

import vtl

fn test_repeat_flattens_by_default() {
	t := vtl.from_array([1, 2, 3, 4], [2, 2])!
	repeated := vtl.repeat(t, 2)!
	assert repeated.shape == [8]
	assert repeated.to_array() == [1, 1, 2, 2, 3, 3, 4, 4]
}

fn test_repeat_along_axes_and_on_strided_views() {
	t := vtl.from_array([1, 2, 3, 4], [2, 2])!
	assert vtl.repeat_axis(t, 2, 0)!.to_array() == [1, 2, 1, 2, 3, 4, 3, 4]
	assert vtl.repeat_axis(t, 2, -1)!.to_array() == [1, 1, 2, 2, 3, 3, 4, 4]
	view := t.slice([0, 2], []int{})!
	assert vtl.repeat_axis(view, 2, 1)!.to_array() == [1, 1, 2, 2, 3, 3, 4, 4]
}

fn test_repeat_zero_and_invalid_repeats_or_axis() {
	t := vtl.from_1d([1, 2])!
	assert vtl.repeat(t, 0)!.shape == [0]
	assert vtl.repeat_axis(t, 0, 0)!.shape == [0]
	if _ := vtl.repeat(t, -1) {
		assert false, 'repeat must reject negative counts'
	} else {
		assert true
	}
	if _ := vtl.repeat_axis(t, 2, 1) {
		assert false, 'repeat must reject out-of-range axes'
	} else {
		assert true
	}
}
