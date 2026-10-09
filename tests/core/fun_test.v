module core

import vtl

fn test_apply() {
	a := vtl.from_1d([1, 2, 3, 4])!
	b := vtl.from_1d([0, 1, 2, 3])!
	mut c := a.add(b)!
	c.apply(fn (x int, i []int) int {
		return x * 2
	})
	expected := vtl.from_1d([2, 6, 10, 14])!
	assert c.array_equal(expected)
}

fn test_map() {
	a := vtl.from_1d([1, 2, 3, 4])!
	b := vtl.from_1d([0, 1, 2, 3])!
	c := a.add(b)!
	d := c.map(fn (x int, i []int) int {
		return x * 2
	})
	expected := vtl.from_1d([2, 6, 10, 14])!
	assert d.array_equal(expected)
}

fn test_map_values_and_map_pair_values() ! {
	a := vtl.from_2d([[1, 2], [3, 4]])!
	doubled := a.map_values(fn (value int) int { return value * 2 })
	expected_doubled := vtl.from_2d([[2, 4], [6, 8]])!
	assert doubled.array_equal(expected_doubled)

	b := vtl.from_1d([10, 20])!
	summed := a.map_pair_values(b, fn (x int, y int) int { return x + y })!
	expected_sum := vtl.from_2d([[11, 22], [13, 24]])!
	assert summed.array_equal(expected_sum)

	transposed := a.transpose([1, 0])!
	incremented := transposed.map_values(fn (value int) int { return value + 1 })
	expected_transposed := vtl.from_2d([[2, 4], [3, 5]])!
	assert incremented.array_equal(expected_transposed)
}

fn test_reduce() {
	a := vtl.from_1d([1, 2, 3, 4])!
	b := a.reduce(0, fn (acc int, x int, i []int) int {
		return acc + x
	})
	assert b == 10
}

fn test_napply() {
	a := vtl.from_1d([1, 2, 3, 4])!
	b := vtl.from_1d([0, 1, 2, 3])!
	mut c := a.add(b)! // [1, 3, 5, 7]
	c.napply([a, b], fn (xs []int, i []int) int {
		return xs[0] * xs[1] - xs[2]
	})!
	expected := vtl.from_1d([1, 5, 13, 25])!
	assert c.array_equal(expected)
}

fn test_nmap() {
	a := vtl.from_1d([1, 2, 3, 4])!
	b := vtl.from_1d([0, 1, 2, 3])!
	c := a.add(b)! // [1, 3, 5, 7]
	d := c.nmap([a, b], fn (xs []int, i []int) int {
		return xs[0] * xs[1] - xs[2]
	})!
	expected := vtl.from_1d([1, 5, 13, 25])!
	assert d.array_equal(expected)
}

fn test_nreduce() {
	a := vtl.from_1d([1, 2, 3, 4])!
	b := vtl.from_1d([0, 1, 2, 3])!
	c := a.add(b)! // [1, 3, 5, 7]
	d := c.nreduce([a, b], 0, fn (acc int, xs []int, i []int) int {
		return acc + xs[0] * xs[1] - xs[2]
	})!
	assert d == 44
}

fn test_reshape_tensor_with_know_dim() {
	values := []int{len: 27, init: index}
	a := vtl.from_array(values, [3, 3, 3])!
	b := vtl.from_array(values, [9, 3])!
	assert a.reshape(b.shape)!.array_equal(b)
}

fn test_reshape_tensor_with_unknow_dim() {
	values := []int{len: 27, init: index}
	a := vtl.from_array(values, [3, 3, 3])!
	b := vtl.from_array(values, [9, 3])!
	c := a.reshape([-1, 3])!
	assert c.array_equal(b)
}

fn test_reshape_transposed_tensor_preserves_row_major_logical_order() ! {
	tensor := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	transposed := tensor.transpose([1, 0])!
	reshaped := transposed.reshape([6])!
	assert reshaped.to_array() == [1, 4, 2, 5, 3, 6]
	reshaped.set_nth(0, 99)
	assert tensor.get([0, 0]) == 1
}

fn test_reshape_column_major_tensor_preserves_row_major_logical_order() ! {
	tensor := vtl.from_2d([[1, 2, 3], [4, 5, 6]], memory: .col_major)!
	reshaped := tensor.reshape([3, 2])!
	assert reshaped.to_array() == [1, 2, 3, 4, 5, 6]
	reshaped.set_nth(0, 99)
	assert tensor.get([0, 0]) == 1
}

fn test_cant_reshape_tensor_with_know_dim() {
	values := []int{len: 27, init: index}
	a := vtl.from_array(values, [3, 3, 3])!
	assert a.shape == [3, 3, 3]
	if b := a.reshape([10, 10]) {
		assert false
	} else {
		assert true
	}
}

fn test_t() {
	t := vtl.from_2d([[6.0, 4, 24], [1.0, -9, 8]])!
	tt := t.t()!
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
	assert tt.get([0, 1]) == 1.0
	assert tt.get([2, 0]) == 24.0
	assert tt.get([2, 1]) == 8.0
}

fn test_ones_t() {
	t := vtl.ones[f64]([2, 3])
	tt := t.t()!
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
	assert tt.get([1, 1]) == 1.0
}

fn test_transpose() {
	t := vtl.ones[f64]([2, 3])
	tt := t.transpose([1, 0])!
	assert t.shape == [2, 3]
	assert tt.shape == [3, 2]
}

fn test_moveaxis() {
	t := vtl.from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
		20, 21, 22, 23],
		[2, 3, 4])!
	moved := t.moveaxis([0], [1])!
	assert moved.shape == [3, 2, 4]
	assert moved.get([2, 1, 3]) == 23
	assert t.get([1, 2, 3]) == 23

	moved_multiple := t.moveaxis([0, -1], [-1, 0])!
	assert moved_multiple.shape == [4, 3, 2]
	assert moved_multiple.get([3, 2, 1]) == 23

	mut base := vtl.from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23],
		[2, 3, 4])!
	mut view := base.moveaxis([0], [1])!
	view.set([2, 1, 3], -1)
	assert base.get([1, 2, 3]) == -1
}

fn test_moveaxis_rejects_invalid_axes() {
	t := vtl.ones[int]([2, 3, 4])
	if _ := t.moveaxis([0], [0, 1]) {
		assert false, 'moveaxis must reject mismatched axis lists'
	} else {
		assert true
	}
	if _ := t.moveaxis([0, 0], [1, 2]) {
		assert false, 'moveaxis must reject duplicate source axes'
	} else {
		assert true
	}
	if _ := t.moveaxis([0, 1], [2, 2]) {
		assert false, 'moveaxis must reject duplicate destination axes'
	} else {
		assert true
	}
	if _ := t.moveaxis([3], [0]) {
		assert false, 'moveaxis must reject out-of-range axes'
	} else {
		assert true
	}
}

fn test_rollaxis() {
	t := vtl.ones[int]([2, 3, 4])
	assert t.rollaxis(2, 0)!.shape == [4, 2, 3]
	assert t.rollaxis(0, 3)!.shape == [3, 4, 2]
	assert t.rollaxis(-1, 1)!.shape == [2, 4, 3]
	if _ := t.rollaxis(3, 0) {
		assert false, 'rollaxis must reject out-of-range axes'
	} else {
		assert true
	}
	if _ := t.rollaxis(0, 4) {
		assert false, 'rollaxis must reject out-of-range start positions'
	} else {
		assert true
	}
}

fn test_slice() {
	a := vtl.from_array([0.0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3])!
	slice := a.slice([0])!
	expected := vtl.from_array([0.0, 1, 2], [3])!
	assert slice.array_equal(expected)
}

fn test_slice_implicit() {
	a := vtl.from_array([0.0, 1, 2, 3], [2, 2])!
	slice := a.slice([]int{}, [1])!
	expected := vtl.from_array([1.0, 3], [2])!
	assert slice.array_equal(expected)
}

fn test_negative_slice() {
	a := vtl.from_array([1.0, 2, 3], [3])!
	slice := a.slice([0, 3, -1])!
	expected := vtl.from_array([3.0, 2, 1], [3])!
	assert slice.array_equal(expected)
}

fn test_slice_hilo() {
	t := vtl.from_array([1.0, 2, 3, 4], [2, 2])!
	slice := t.slice_hilo([0], [2])!
	assert t.array_equal(slice)
}
