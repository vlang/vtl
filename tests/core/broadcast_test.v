// vtest flaky: true
module core

import vtl

fn test_broadcast_column() {
	m := vtl.from_array([1.0, 2.0, 3.0], [3, 1])!
	b := m.broadcast_to([3, 3])!
	expected := vtl.from_array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], [3, 3])!
	assert b.array_equal(expected)
}

fn test_broadcastable_same_shape() {
	m := vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!
	shape := m.broadcastable(m)!
	assert m.shape == shape
}

fn test_broadcastable_different_shape1() {
	a := vtl.zeros[f64]([8, 1, 6, 1])
	b := vtl.zeros[f64]([7, 1, 5])
	shape := a.broadcastable(b)!
	assert shape == [8, 7, 6, 5]
}

fn test_broadcastable_different_shape2() {
	a := vtl.from_1d([0, 1, 2])!
	expected := vtl.from_2d([[0, 1, 2], [0, 1, 2], [0, 1, 2]])!
	result := a.broadcast_to([3, 3])!
	assert result.array_equal(expected)
}

fn test_cant_broadcast() {
	a := vtl.from_1d([0, 1, 2])!
	if _ := a.broadcast_to([3, 5]) {
		assert false
	} else {
		assert true
	}
}

fn test_broadcast_eachother_1() {
	a := vtl.from_array([0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3])!
	b := vtl.from_1d([0, 1, 2])!
	ra, rb := vtl.broadcast2(a, b)!
	assert ra.shape == rb.shape
}

fn test_cant_broadcast_eachother_1() {
	a := vtl.from_array([0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3])!
	b := vtl.from_1d([0, 1, 2, 4])!
	if _, _ := vtl.broadcast2(a, b) {
		assert false
	} else {
		assert true
	}
}

fn test_broadcast_empty_dimensions_follow_numpy_shape_rules() ! {
	empty := vtl.from_array([]int{}, [0, 3])!
	rows := vtl.from_array([1, 2, 3], [1, 3])!
	left, right := vtl.broadcast2(empty, rows)!
	assert left.shape == [0, 3]
	assert right.shape == [0, 3]
	assert left.size == 0
	assert right.size == 0
	assert empty.broadcastable(rows)! == [0, 3]

	conflicting := vtl.from_array([1, 2], [2, 1])!
	if _ := empty.broadcastable(conflicting) {
		assert false
	} else {
		assert err.msg().contains('not broadcastable')
	}
}

fn test_broadcast_n_rejects_empty_and_incompatible_inputs() {
	empty := []&vtl.Tensor[int]{}
	if _ := vtl.broadcast_n[int](empty) {
		assert false
	} else {
		assert err.msg().contains('at least one tensor')
	}

	a := vtl.from_1d([1, 2])!
	b := vtl.from_1d([1, 2, 3])!
	if _ := vtl.broadcast_n[int]([a, b]) {
		assert false
	} else {
		assert err.msg().contains('not broadcastable')
	}
}

fn test_broadcast_to_rejects_lower_rank_and_negative_dimensions() {
	tensor := vtl.from_array([1, 2, 3, 4], [2, 2])!
	if _ := tensor.broadcast_to([2]) {
		assert false
	} else {
		assert err.msg().contains('target rank is smaller')
	}
	if _ := tensor.broadcast_to([2, -1]) {
		assert false
	} else {
		assert err.msg().contains('negative dimensions')
	}
}
