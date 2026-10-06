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

fn test_tile_repeats_blocks_and_aligns_repetitions_from_the_right() {
	t := vtl.from_array([1, 2, 3, 4], [2, 2])!
	assert vtl.tile(t, [2, 1])!.to_array() == [1, 2, 3, 4, 1, 2, 3, 4]
	assert vtl.tile(t, [2])!.shape == [2, 4]
	assert vtl.tile(t, [2])!.to_array() == [1, 2, 1, 2, 3, 4, 3, 4]
	assert vtl.tile(vtl.from_1d([5, 6])!, [2, 1])!.shape == [2, 2]
	assert vtl.tile(vtl.from_1d([5, 6])!, [2, 1])!.to_array() == [5, 6, 5, 6]
}

fn test_tile_zero_repetitions_and_rejects_negative_repetitions() {
	t := vtl.from_1d([1, 2])!
	assert vtl.tile(t, [0])!.shape == [0]
	if _ := vtl.tile(t, [-1]) {
		assert false, 'tile must reject negative repetitions'
	} else {
		assert true
	}
}

fn test_rot90_and_custom_axes() {
	t := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
	assert vtl.rot90(t)!.shape == [3, 2]
	assert vtl.rot90(t)!.to_array() == [3, 6, 2, 5, 1, 4]
	assert vtl.rot90_k(t, -1)!.to_array() == [4, 1, 5, 2, 6, 3]
	assert vtl.rot90_k(t, 2)!.to_array() == [6, 5, 4, 3, 2, 1]
	assert vtl.rot90_axes(vtl.from_array([1, 2, 3, 4, 5, 6], [2, 1, 3])!, 1,
		[0, 2])!.shape == [3, 1, 2]
}

fn test_rot90_rejects_invalid_axes() {
	t := vtl.from_1d([1, 2])!
	if _ := vtl.rot90(t) {
		assert false, 'rot90 must reject tensors with fewer than two dimensions'
	} else {
		assert true
	}
}
