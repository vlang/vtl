// vtest flaky: true
module main

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
