import vtl

fn test_broadcast_column() {
	m := vtl.from_varray<f64>([1., 2., 3.], [3, 1])
	b := m.broadcast_to([3, 3])
	expected := vtl.from_varray<f64>([1., 1., 1., 2., 2., 2., 3., 3., 3.], [3, 3])
	assert vtl.tensor_to_varray<f64>(b) == vtl.tensor_to_varray<f64>(expected)
}

fn test_broadcastable_same_shape() {
	m := vtl.from_varray<f64>([1., 2., 3., 4.], [2, 2])
	shape := vtl.broadcastable(m, m)
	assert m.shape == shape
}

fn test_broadcastable_different_shape() {
	a := vtl.new_tensor(shape: [8, 1, 6, 1])
	b := vtl.new_tensor(shape: [7, 1, 5])
	shape := vtl.broadcastable(a, b)
	assert shape == [8, 7, 6, 5]
}
