module vtl

fn test_broadcast_column() {
	m := from_array([1.0, 2.0, 3.0], [3, 1]) or { panic(@FN + ' failed') }
	b := m.broadcast_to([3, 3]) or { panic(@FN + ' failed') }
	expected := from_array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], [3, 3]) or {
		panic(@FN + ' failed')
	}
	assert b.array_equiv(expected)
}

fn test_broadcastable_same_shape() {
	m := from_array([1.0, 2.0, 3.0, 4.0], [2, 2]) or { panic(@FN + ' failed') }
	shape := m.broadcastable(m) or { panic(@FN + ' failed') }
	assert m.shape == shape
}

fn test_broadcastable_different_shape() {
	a := zeros<f64>([8, 1, 6, 1])
	b := zeros<f64>([7, 1, 5])
	shape := a.broadcastable(b) or { panic(@FN + ' failed') }
	assert shape == [8, 7, 6, 5]
}
