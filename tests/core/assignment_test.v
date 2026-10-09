module core

import vtl

fn test_set() {
	mut t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5])!
	t.set([0, 0], 16)
	t.set([0, 1], 17)
	t.set([0, 2], 18)
	t.set([0, 3], 19)
	t.set([0, 4], 20)
	t.set([1, 0], 11)
	t.set([1, 1], 12)
	t.set([1, 2], 13)
	t.set([1, 3], 14)
	t.set([1, 4], 15)
	expected := vtl.from_array([16, 17, 18, 19, 20, 11, 12, 13, 14, 15], [2, 5])!
	assert t.array_equal(expected)
}

fn test_set_nth() {
	mut t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5])!
	t.set_nth(0, 16)
	t.set_nth(1, 17)
	t.set_nth(2, 18)
	t.set_nth(3, 19)
	t.set_nth(4, 20)
	t.set_nth(5, 11)
	t.set_nth(6, 12)
	t.set_nth(7, 13)
	t.set_nth(8, 14)
	t.set_nth(9, 15)
	expected := vtl.from_array([16, 17, 18, 19, 20, 11, 12, 13, 14, 15], [2, 5])!
	assert t.array_equal(expected)
}

fn test_fill() {
	mut t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5])!
	t.fill(-1)
	expected := vtl.from_array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [2, 5])!
	assert t.array_equal(expected)
}

fn test_assign() {
	mut t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5])!
	a := vtl.from_array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [2, 5])!
	t.assign(a)!
	assert t.array_equal(a)
	b := vtl.from_1d([21, 22, 23, 24, 25])!
	t.assign(b)!
	expected := vtl.from_array([21, 22, 23, 24, 25, 21, 22, 23, 24, 25], [2, 5])!
	assert t.array_equal(expected)
}

fn test_put_flat_indices_repeat_values_and_preserve_logical_view_order() ! {
	mut target := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
	indices := vtl.from_1d([0, -1, 0])!
	values := vtl.from_1d([10, 20])!
	target.put(indices, values)!
	assert target.to_array() == [10, 2, 3, 4, 5, 20]

	mut matrix := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
	mut transposed := matrix.transpose([1, 0])!
	transposed.put(vtl.from_1d([1])!, vtl.from_1d([99])!)!
	assert matrix.to_array() == [1, 2, 3, 99, 5, 6]
}

fn test_put_flat_indices_snapshot_aliased_indices() ! {
	mut target := vtl.from_1d([1, 0, 2])!
	indices := target.view()
	updates := vtl.from_1d([7, 3, 0])!
	target.put(indices, updates)!
	assert target.to_array() == [3, 7, 0]
}

fn test_put_flat_indices_snapshot_aliased_values() ! {
	mut target := vtl.from_1d([1, 2, 3, 4])!
	indices := vtl.from_1d([3, 2, 1, 0])!
	values := target.view()
	target.put(indices, values)!
	assert target.to_array() == [4, 3, 2, 1]
}

fn test_put_flat_indices_support_wrap_and_clip_modes() ! {
	mut wrapped := vtl.zeros[int]([4])
	wrapped.put_with_mode(vtl.from_1d([-1, 4, -5])!, vtl.from_1d([1, 2, 3])!, .wrap)!
	assert wrapped.to_array() == [2, 0, 0, 3]

	mut clipped := vtl.zeros[int]([4])
	clipped.put_with_mode(vtl.from_1d([-4, 4])!, vtl.from_1d([7, 8])!, .clip)!
	assert clipped.to_array() == [7, 0, 0, 8]
}

fn test_put_flat_indices_validates_before_mutating() ! {
	mut target := vtl.from_1d([1, 2, 3])!
	if _ := target.put(vtl.from_1d([0, 3])!, vtl.from_1d([9, 8])!) {
		assert false, 'put must reject out-of-range indices'
	}
	assert target.to_array() == [1, 2, 3]

	if _ := target.put(vtl.from_1d([1])!, vtl.from_1d[int]([])!) {
		assert false, 'put must reject empty values for non-empty indices'
	}
	assert target.to_array() == [1, 2, 3]

	target.put(vtl.from_1d[int]([])!, vtl.from_1d[int]([])!)!
}

fn test_put_along_axis_supports_negative_indices_and_last_duplicate_write() ! {
	mut target := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
	indices := vtl.from_array([-1, 1, 0, 0], [2, 2])!
	updates := vtl.from_array([10, 20, 30, 40], [2, 2])!
	target.put_along_axis(indices, updates, -1)!
	expected := vtl.from_array([1, 20, 10, 40, 5, 6], [2, 3])!
	assert target.array_equal(expected)
}

fn test_scatter_add_accumulates_duplicate_indices() ! {
	mut target := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
	indices := vtl.from_array([1, 1, 0, 0], [2, 2])!
	updates := vtl.from_array([10, 20, 30, 40], [2, 2])!
	target.scatter_add(indices, updates, 1)!
	expected := vtl.from_array([1, 32, 3, 74, 5, 6], [2, 3])!
	assert target.array_equal(expected)
}

fn test_put_along_axis_accepts_partial_non_axis_dimensions() ! {
	mut target := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
	indices := vtl.from_array([2, 0], [1, 2])!
	updates := vtl.from_array([9, 8], [1, 2])!
	target.put_along_axis(indices, updates, 1)!
	expected := vtl.from_array([8, 2, 9, 4, 5, 6], [2, 3])!
	assert target.array_equal(expected)
}

fn test_indexed_updates_reject_invalid_shapes_axes_and_indices() {
	mut target := vtl.zeros[int]([2, 3])
	indices := vtl.from_array([0, 1], [1, 2])!
	updates := vtl.ones[int]([1, 2])
	if _ := target.put_along_axis(indices, updates, 2) {
		assert false, 'put_along_axis must reject an invalid axis'
	}
	if _ := target.scatter_add(indices, vtl.ones[int]([2, 2]), 1) {
		assert false, 'scatter_add must reject a shape mismatch'
	}
	bad_indices := vtl.from_array([0, 3], [1, 2])!
	if _ := target.put_along_axis(bad_indices, updates, 1) {
		assert false, 'put_along_axis must reject an out-of-range index'
	}
	assert target.array_equal(vtl.zeros[int]([2, 3]))
}
