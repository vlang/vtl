module vtl

fn test_t() {
	t := from_2d([[6.0, 4, 24], [1.0, -9, 8]]) or { panic(@FN + ' failed') }
	tt := t.t() or { panic(@FN + ' failed') }
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
	assert tt.get([0, 1]) == 1.0
	assert tt.get([2, 0]) == 24.0
	assert tt.get([2, 1]) == 8.0
}

fn test_ones_t() {
	t := ones<f64>([2, 3])
	tt := t.t() or { panic(@FN + ' failed') }
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
	assert tt.get([1, 1]) == 1.0
}

fn test_transpose() {
	t := ones<f64>([2, 3])
	order := irange(0, t.rank())
	tt := t.transpose(order.reverse()) or { panic(@FN + ' failed') }
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
}

fn test_slice() {
	a := from_array([0.0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3]) or { panic(@FN + ' failed') }
	slice := a.slice([0]) or { panic(@FN + ' failed') }
	expected := from_array([0.0, 1, 2], [3]) or { panic(@FN + ' failed') }
	assert slice.array_equal(expected)
}

fn test_slice_implicit() {
	a := from_array([0.0, 1, 2, 3], [2, 2]) or { panic(@FN + ' failed') }
	slice := a.slice([]int{}, [1]) or { panic(@FN + ' failed') }
	expected := from_array([1.0, 3], [2]) or { panic(@FN + ' failed') }
	assert slice.array_equal(expected)
}

fn test_negative_slice() {
	a := from_array([1.0, 2, 3], [3]) or { panic(@FN + ' failed') }
	slice := a.slice([0, 3, -1]) or { panic(@FN + ' failed') }
	expected := from_array([3.0, 2, 1], [3]) or { panic(@FN + ' failed') }
	assert slice.array_equal(expected)
}

fn test_slice_hilo() {
	t := from_array([1.0, 2, 3, 4], [2, 2]) or { panic(@FN + ' failed') }
	slice := t.slice_hilo([0], [2]) or { panic(@FN + ' failed') }
	assert t.array_equal(slice)
}
