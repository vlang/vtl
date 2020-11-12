import vtl.num

fn test_broadcast_to() {
	m := num.from_int_1d([1, 2, 3])
	b := num.broadcast_to(m, [3, 3])
	expected := num.from_int([1, 2, 3, 1, 2, 3, 1, 2, 3], [3, 3])
	assert num.allclose(b, expected)
}

fn test_broadcast_column() {
	m := num.from_int([1, 2, 3], [3, 1])
	b := num.broadcast_to(m, [3, 3])
	expected := num.from_int([1, 1, 1, 2, 2, 2, 3, 3, 3], [3, 3])
	assert num.allclose(b, expected)
}

fn test_broadcastable_same_shape() {
	m := num.from_int([1, 2, 3, 4], [2, 2])
	shape := num.broadcastable(m, m)
	assert num.shape_compare(m.shape, shape)
}

fn test_broadcastable_different_shape() {
	a := num.allocate_cpu([8, 1, 6, 1], 'C')
	b := num.allocate_cpu([7, 1, 5], 'C')
	shape := num.broadcastable(a, b)
	assert num.shape_compare(shape, [8, 7, 6, 5])
}

fn test_as_strided() {
	a := num.from_int_1d([0, 1, 2, 3, 4, 5, 6, 7])
	n := a.strides[0]
	res := num.as_strided(a, [5, 3], [n, n])
	expected := num.from_int([0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6], [5, 3])
	assert num.allclose(res, expected)
}

fn test_expand_dims() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	res := num.expand_dims(a, 1)
	expected := num.from_int([1, 2, 3, 4], [2, 1, 2])
	assert num.allclose(res, expected)
}
