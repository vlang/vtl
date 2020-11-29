module vtl

fn test_new() {
	init := 1.0
	t := new_tensor<f64>({
		shape: [3]
		init: &init
	})
	varray := tensor_to_varray<f64>(t)
	assert varray.len == 3
	assert varray[0] == 1.0
}

fn test_from_varray() {
	arr := [1.0, 2.0]
	shape := [2]
	mut t := from_varray(arr, shape)
	varray := tensor_to_varray<f64>(t)
	assert varray == arr
}

fn test_from_1d() {
	arr := [1.0, 2.0]
	mut t := from_1d<f64>(arr)
	varray := tensor_to_varray<f64>(t)
	assert varray == arr
}

fn test_from_2d() {
	arr := [[1.0, 2.0]]
	mut t := from_2d<f64>(arr)
	varray := tensor_to_varray<f64>(t)
	assert varray.len == 2
	assert varray[1] == 2.0
}

fn test_eye() {
	res := eye<f64>(3, 3, 0)
	expected := from_varray([1., 0., 0., 0., 1., 0., 0., 0., 1.], [3, 3])
	assert tensor_to_varray<f64>(res) == tensor_to_varray<f64>(expected)
}

fn test_eye_different_shape() {
	res := eye<f64>(2, 4, 0)
	expected := from_varray([1., 0., 0., 0., 0., 1., 0., 0.], [2, 4])
	assert tensor_to_varray<f64>(res) == tensor_to_varray<f64>(expected)
}

fn test_eye_offset() {
	res := eye<f64>(3, 3, 1)
	expected := from_varray([0., 1., 0., 0., 0., 1., 0., 0., 0.], [3, 3])
	assert tensor_to_varray<f64>(res) == tensor_to_varray<f64>(expected)
}

fn test_identity() {
	res := identity<f64>(3)
	expected := from_varray([1., 0., 0., 0., 1., 0., 0., 0., 1.], [3, 3])
	assert tensor_to_varray<f64>(res) == tensor_to_varray<f64>(expected)
}

fn test_zeros() {
	mut t := zeros<f64>([3])
	varray := tensor_to_varray<f64>(t)
	assert varray == [0.0, 0.0, 0.0]
}

fn test_ones() {
	mut t := ones<f64>([3])
	varray := tensor_to_varray<f64>(t)
	assert varray == [1.0, 1.0, 1.0]
}

fn test_full() {
	mut t := full<f64>([3], 3.0)
	varray := tensor_to_varray<f64>(t)
	assert varray == [3.0, 3.0, 3.0]
}
