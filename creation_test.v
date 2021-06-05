module vtl

fn test_new() {
	t := new_tensor(shape: [3], init: 1.0)
	assert t.size() == 3
	assert t.get([0]) as f64 == 1.
}

fn test_from_varray() {
	arr := [1.0, 2.0]
	shape := [2]
	mut t := from_varray<f64>(arr, shape)
	assert t.get([0]) as f64 == 1.
	assert t.get([1]) as f64 == 2.
}

fn test_from_1d() {
	arr := [1.0, 2.0]
	mut t := from_1d<f64>(arr)
	assert t.get([0]) as f64 == 1.
	assert t.get([1]) as f64 == 2.
}

fn test_from_2d() {
	arr := [[1.0, 2.0]]
	mut t := from_2d<f64>(arr)
	assert t.get([0, 0]) as f64 == 1.
	assert t.get([0, 1]) as f64 == 2.
}

fn test_eye() {
	res := eye(3, 3, 0)
	expected := from_varray<f64>([1.0, 0., 0., 0., 1., 0., 0., 0., 1.], [3, 3])
	assert res.equal(expected)
}

fn test_eye_different_shape() {
	res := eye(2, 4, 0)
	expected := from_varray<f64>([1.0, 0., 0., 0., 0., 1., 0., 0.], [2, 4])
	assert res.equal(expected)
}

fn test_eye_offset() {
	res := eye(3, 3, 1)
	expected := from_varray<f64>([0., 1., 0., 0., 0., 1., 0., 0., 0.], [3, 3])
	assert res.equal(expected)
}

fn test_identity() {
	res := identity(3)
	expected := from_varray<f64>([1.0, 0., 0., 0., 1., 0., 0., 0., 1.], [3, 3])
	assert res.equal(expected)
}

fn test_zeros() {
	mut t := zeros([3])
	assert t.get([0]) as f64 == 0.
	assert t.get([1]) as f64 == 0.
	assert t.get([2]) as f64 == 0.
}

fn test_ones() {
	mut t := ones([3])
	assert t.get([0]) as f64 == 1.
	assert t.get([1]) as f64 == 1.
	assert t.get([2]) as f64 == 1.
}

fn test_full() {
	mut t := full([3], f64(3.0))
	assert t.get([0]) as f64 == 3.
	assert t.get([1]) as f64 == 3.
	assert t.get([2]) as f64 == 3.
}
