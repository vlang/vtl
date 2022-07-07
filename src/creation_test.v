module vtl

fn test_eye() {
	res := eye<f64>(3, 3, 0)
	expected := from_array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3])
	for i in 0 .. res.size() {
		println('${res.get([0, i])} ${expected.get([0, i])}')
		assert res.get([0, i]) == expected.get([0, i])
	}
}

fn test_eye_different_shape() {
	res := eye<f64>(2, 4, 0)
	expected := from_array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [2, 4])
	for i in 0 .. res.size() {
		println('${res.get([0, i])} ${expected.get([0, i])}')
		assert res.get([0, i]) == expected.get([0, i])
	}
}

fn test_eye_offset() {
	res := eye<f64>(3, 3, 1)
	expected := from_array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [3, 3])
	for i in 0 .. res.size() {
		println('${res.get([0, i])} ${expected.get([0, i])}')
		assert res.get([0, i]) == expected.get([0, i])
	}
}

fn test_identity() {
	res := identity<f64>(3)
	expected := from_array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3])
	for i in 0 .. res.size() {
		println('${res.get([0, i])} ${expected.get([0, i])}')
		assert res.get([0, i]) == expected.get([0, i])
	}
}

fn test_zeros() {
	mut t := zeros<f64>([3])
	assert t.get([0]) == 0.0
	assert t.get([1]) == 0.0
	assert t.get([2]) == 0.0
}

fn test_ones() {
	mut t := ones<f64>([3])
	assert t.get([0]) == 1.0
	assert t.get([1]) == 1.0
	assert t.get([2]) == 1.0
}

fn test_full() {
	mut t := full<f64>([3], 3.0)
	assert t.get([0]) == 3.0
	assert t.get([1]) == 3.0
	assert t.get([2]) == 3.0
}
