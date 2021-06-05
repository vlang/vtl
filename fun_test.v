module vtl

fn test_t() {
	t := from_2d([[6., 4, 24], [1., -9, 8]])
	tt := t.t()
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
	assert tt.get([0, 1]) as f64 == 1.
	assert tt.get([2, 0]) as f64 == 24.
	assert tt.get([2, 1]) as f64 == 8.
}

fn test_ones_t() {
	t := ones([2, 3])
	tt := t.t()
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
	assert tt.get([1, 1]) as f64 == 1.
}

fn test_transpose() {
	t := ones([2, 3])
	order := irange(0, t.rank())
	tt := t.transpose(order.reverse())
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
}
