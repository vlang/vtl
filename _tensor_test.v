module vtl

fn test_empty_shape() {
	t := new_tensor<f64>({
		shape: []
	})
	assert t.rank() == 0
	assert t.strides.len == 1
	assert t.strides[0] == 1
	assert t.size() == 1
}

fn test_rowmajor() {
	t := new_tensor<f64>({
		shape: [2, 2]
	})
	assert t.size() == 4
	assert t.is_contiguous()
	assert !t.is_colmajor()
}

fn test_colmajor() {
	t := new_tensor<f64>({
		shape: [2, 2]
		memory: .colmajor
	})
	assert t.size() == 4
	assert t.is_contiguous()
	assert t.is_colmajor()
}

fn test_get() {
	arr := [1., 2., 3., 4.]
	shape := [2, 2]
	t := from_varray<f64>(arr, shape)
	val := t.get([1, 1])
	mut a := *(&f64(val))
	assert a == 4.0
}

fn test_set() {
	arr := [1., 2., 3., 4.]
	shape := [2, 2]
	newval := 8.0
	mut t := from_varray<f64>(arr, shape)
	t.set([1, 1], &newval)
	val := t.get([1, 1])
	mut a := *(&f64(val))
	assert a == 8.0
}
