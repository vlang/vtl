module vtl

fn test_empty_shape() {
	t := new_tensor(shape: [])
	assert t.rank() == 0
	assert t.strides.len == 1
	assert t.strides[0] == 1
	assert t.size() == 1
}

fn test_rowmajor() {
	t := new_tensor(shape: [2, 2])
	assert t.size() == 4
	assert t.is_contiguous()
	assert !t.is_colmajor()
}

fn test_colmajor() {
	t := new_tensor(shape: [2, 2], memory: .colmajor)
	assert t.size() == 4
	assert t.is_contiguous()
	assert t.is_colmajor()
}

fn test_get() {
	arr := [1., 2., 3., 4.]
	shape := [2, 2]
	t := from_varray<f64>(arr, shape)
	val := t.get([1, 1]) as f64
	assert val == 4.0
}

fn test_set() {
	arr := [1., 2., 3., 4.]
	shape := [2, 2]
	mut t := from_varray<f64>(arr, shape)
	t.set([1, 1], f64(8.0))
	val := t.get([1, 1]) as f64
	assert val == 8.0
}

fn test_transpose() {
	t := from_varray<f64>([1.0, 2., 3., 4.], [2, 2])
	v1 := t.transpose([1, 0])
	v2 := t.t()
	v3 := t.swapaxes(1, 0)
	v1_array := tensor_to_varray<f64>(v1)
	v2_array := tensor_to_varray<f64>(v2)
	v3_array := tensor_to_varray<f64>(v3)
	assert v1_array == v2_array
	assert v2_array == v3_array
}

fn test_as_type() {
	// create tensor of integers
	t := new_tensor(shape: [2, 2], init: int(1))
	tf64 := tensor_as_type<f64>(t)
	assert t.get([1, 1]) as int == 1
	assert tensor_to_varray<f64>(t) == tensor_to_varray<f64>(tf64)
}
