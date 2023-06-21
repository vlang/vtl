module vtl

fn test_from_array() {
	t := from_array([1.0, 2.0, 3.0], [3])!
	assert t.size() == 3
	assert t.get([0]) == 1.0
	assert t.get([1]) == 2.0
	assert t.get([2]) == 3.0
}

fn test_tensor() {
	t := tensor(1.0, [3])
	assert t.size() == 3
	assert t.get([0]) == 1.0
	assert t.get([1]) == 1.0
	assert t.get([2]) == 1.0
}

fn test_tensor_like() {
	t := tensor(1.0, [3])
	mut t2 := tensor_like(t)
	t2.fill(2.0)
	assert t2.size() == 3
	assert t2.get([0]) == 2.0
	assert t2.get([1]) == 2.0
	assert t2.get([2]) == 2.0
}

fn test_tensor_like_with_shape() {
	t := tensor(1.0, [3])
	mut t2 := tensor_like_with_shape(t, [2, 3])
	t2.fill(1.0)
	assert t2.size() == 6
	assert t2.get([0, 0]) == 1.0
	assert t2.get([0, 1]) == 1.0
	assert t2.get([0, 2]) == 1.0
	assert t2.get([1, 0]) == 1.0
	assert t2.get([1, 1]) == 1.0
	assert t2.get([1, 2]) == 1.0
}

fn test_tensor_like_with_shape_and_strides() {
	t := tensor(1.0, [3])
	mut t2 := tensor_like_with_shape_and_strides(t, [2, 3], [3, 1])
	t2.fill(1.0)
	assert t2.size() == 6
	assert t2.get([0, 0]) == 1.0
	assert t2.get([0, 1]) == 1.0
	assert t2.get([0, 2]) == 1.0
	assert t2.get([1, 0]) == 1.0
	assert t2.get([1, 1]) == 1.0
	assert t2.get([1, 2]) == 1.0
}
