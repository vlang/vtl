import vtl.num

fn test_empty_shape() {
	a := num.allocate_cpu([], 'C')
	assert a.shape.len == 0
	assert a.strides.len == 1
	assert a.strides[0] == 1
	assert a.size == 1
}

fn test_shape_contig() {
	a := num.allocate_cpu([2, 2], 'C')
	assert a.size == 4
	assert num.shape_compare(a.strides, [2, 1])
	assert a.flags.contiguous
	assert !a.flags.fortran
}

fn test_shape_fortran() {
	a := num.allocate_cpu([2, 2], 'F')
	assert a.size == 4
	assert num.shape_compare(a.strides, [1, 2])
	assert a.flags.fortran
	assert !a.flags.contiguous
}

fn test_get() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	assert a.get([1, 1]) == 4.0
}

fn test_set() {
	mut a := num.allocate_cpu([3, 3], 'C')
	a.set([1, 1], 3.0)
	expected := num.from_int([0, 0, 0, 0, 3, 0, 0, 0, 0], [3, 3])
	assert num.allclose(a, expected)
}

fn test_slice() {
	a := num.from_int([0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3])
	slice := a.slice([[0]])
	expected := num.from_int([0, 1, 2], [3])
	assert num.allclose(expected, slice)
}

fn test_slice_implicit() {
	a := num.from_int([0, 1, 2, 3], [2, 2])
	slice := a.slice([[]int{}, [1]])
	expected := num.from_int([1, 3], [2])
	assert num.allclose(slice, expected)
}

fn test_negative_slice() {
	a := num.from_int([1, 2, 3], [3])
	slice := a.slice([[0, 3, -1]])
	expected := num.from_int([3, 2, 1], [3])
	assert num.allclose(slice, expected)
}

fn test_slice_hilo() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	slice := a.slice_hilo([0], [2])
	assert num.allclose(a, slice)
}

fn test_slice_flags() {
	a := num.from_int([1, 2, 3], [3])
	assert a.flags.owndata
	slice := a.slice([[]int{}])
	assert !slice.flags.owndata
}

fn test_assign_broadcast() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	slice := a.slice([[0]])
	a.assign(slice)
	expected := num.from_int([1, 2, 1, 2], [2, 2])
	assert num.allclose(expected, a)
}

fn test_assign_same_shape() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	slice := a.slice([[0]])
	a.slice([[1]]).assign(slice)
	expected := num.from_int([1, 2, 1, 2], [2, 2])
	assert num.allclose(expected, a)
}

fn test_fill() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	a.fill(10)
	expected := num.from_int([10, 10, 10, 10], [2, 2])
	assert num.allclose(expected, a)
}

fn test_copy() {
	a := num.from_int([1, 2, 3], [3])
	d := a.copy('C')
	assert num.allclose(d, a)
}

fn test_copy_no_update() {
	a := num.from_int([1, 2, 3], [3])
	d := a.copy('C')
	d.fill(2)
	assert !num.allclose(d, a)
}

fn test_view() {
	a := num.from_int([1, 2, 3], [3])
	d := a.view()
	assert num.allclose(a, d)
	assert !d.flags.owndata
}

fn test_diagonal() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	d := a.diagonal()
	expected := num.from_int([1, 4], [2])
	assert num.allclose(d, expected)
}

fn test_reshape() {
	a := num.from_int_1d([1, 2, 3, 4])
	res := a.reshape([2, 2])
	expected := num.from_int([1, 2, 3, 4], [2, 2])
	assert num.allclose(res, expected)
}

fn test_reshape_infer() {
	a := num.from_int_1d([1, 2, 3, 4])
	res := a.reshape([-1, 2])
	expected := num.from_int([1, 2, 3, 4], [2, 2])
	assert num.allclose(res, expected)
}

fn test_transpose() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	v1 := a.transpose([1, 0])
	v2 := a.t()
	v3 := a.swapaxes(1, 0)
	assert num.allclose(v1, v2)
	assert num.allclose(v2, v3)
}

fn test_ravel() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	v1 := a.ravel()
	expected := num.from_int([1, 2, 3, 4], [4])
	assert num.allclose(v1, expected)
}

fn test_elementwise() {
	a := num.from_int_1d([1, 2, 3])
	adde := num.from_int_1d([2, 4, 6])
	subtracte := num.from_int_1d([0, 0, 0])
	divide := num.from_int_1d([1, 1, 1])
	multiply := num.from_int_1d([1, 4, 9])
	assert num.allclose(a.add(a), adde)
	assert num.allclose(a.subtract(a), subtracte)
	assert num.allclose(a.divide(a), divide)
	assert num.allclose(a.multiply(a), multiply)
}
