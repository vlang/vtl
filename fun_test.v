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

// fn test_slice() {
// 	a := from_varray([0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3])
// 	slice := a.slice([0])
// 	expected := from_varray([0, 1, 2], [3])
// 	assert slice.equal(expected)
// }

// fn test_slice_implicit() {
// 	a := from_varray([0, 1, 2, 3], [2, 2])
// 	slice := a.slice([]int{}, [1])
// 	expected := from_varray([1, 3], [2])
// 	assert slice.equal(expected)
// }

// fn test_negative_slice() {
// 	a := from_varray([1, 2, 3], [3])
// 	slice := a.slice([0, 3, -1])
// 	expected := from_varray([3, 2, 1], [3])
// 	assert slice.equal(expected)
// }

fn test_slice_hilo() {
    t := from_varray([1., 2, 3, 4], [2, 2])
    slice := t.slice_hilo([0], [2])
	println(t)
	println(slice)
    assert t.equal(slice)
}
