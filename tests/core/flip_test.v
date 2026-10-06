module core

import vtl

fn test_flip_all_axes_and_negative_axis() {
	t := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
	assert t.flip()!.to_array() == [5, 4, 3, 2, 1, 0]
	assert t.flip(-1)!.to_array() == [2, 1, 0, 5, 4, 3]
	assert t.flip(0)!.to_array() == [3, 4, 5, 0, 1, 2]
}

fn test_flip_supports_repeated_axes_and_strided_views() {
	t := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
	assert t.flip(0, 0)!.to_array() == t.to_array()
	view := t.slice([0, 2], []int{})!
	assert view.flip(1)!.to_array() == [2, 1, 0, 5, 4, 3]
}

fn test_flip_rejects_invalid_axis() {
	t := vtl.ones[int]([2, 3])
	if _ := t.flip(2) {
		assert false, 'flip must reject out-of-range axes'
	} else {
		assert true
	}
}
