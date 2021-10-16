module vtl

fn test_broadcast_column() {
	m := from_array<f64>([1.0, 2.0, 3.0], [3, 1])
	b := m.broadcast_to([3, 3])
	expected := from_array<f64>([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], [3, 3])
	// assert equal<f64>(b, expected)
}

fn test_broadcastable_same_shape() {
	m := from_array<f64>([1.0, 2.0, 3.0, 4.0], [2, 2])
	shape := broadcastable<f64>(m, m)
	assert m.shape == shape
}

fn test_broadcastable_different_shape() {
	a := zeros<f64>([8, 1, 6, 1])
	b := zeros<f64>([7, 1, 5])
	shape := broadcastable<f64>(a, b)
	assert shape == [8, 7, 6, 5]
}
