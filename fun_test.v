import vtl

fn test_t() {
	t := vtl.from_2d([[6., 4, 24], [1., -9, 8]])
	tt := t.t()
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
	assert tt.get([0, 1]) == 1.
	assert tt.get([2, 0]) == 24.
	assert tt.get([2, 1]) == 8.
}

fn test_ones_t() {
	t := vtl.ones<f64>([2, 3])
	tt := t.t()
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
	assert tt.get([1, 1]) == 1.
}

fn test_transpose() {
	t := vtl.ones<f64>([2, 3])
	order := vtl.irange(0, t.rank())
	tt := t.transpose(order.reverse())
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
}

fn test_slice() {
	a := vtl.from_array([0., 1, 2, 3, 4, 5, 6, 7, 8], [3, 3])
	slice := a.slice([0])
	expected := vtl.from_array([0., 1, 2], [3])
	assert slice.equal(expected)
}

fn test_slice_implicit() {
	a := vtl.from_array([0., 1, 2, 3], [2, 2])
	slice := a.slice([]int{}, [1])
	expected := vtl.from_array([1., 3], [2])
	assert slice.equal(expected)
}

fn test_negative_slice() {
	a := vtl.from_array([1., 2, 3], [3])
	slice := a.slice([0, 3, -1])
	expected := vtl.from_array([3., 2, 1], [3])
	assert slice.equal(expected)
}

fn test_slice_hilo() {
	t := vtl.from_array([1., 2, 3, 4], [2, 2])
	slice := t.slice_hilo([0], [2])
	assert t.equal(slice)
}
