module main

import vtl

fn test_empty() {
	mut t := vtl.empty[f64]([3])
	t.fill(1.0)
	assert t.size() == 3
	assert t.get([0]) == 1.0
	assert t.get([1]) == 1.0
	assert t.get([2]) == 1.0
}

fn test_empty_like() {
	mut t := vtl.empty[f64]([3])
	t.fill(1.0)
	assert t.size() == 3
	assert t.get([0]) == 1.0
	assert t.get([1]) == 1.0
	assert t.get([2]) == 1.0
	mut t2 := vtl.empty_like(t)
	assert t2.size() == 3
	assert t2.get([0]) == 0.0
	assert t2.get([1]) == 0.0
	assert t2.get([2]) == 0.0
}

fn test_eye() {
	res := vtl.eye[f64](3, 3, 0)
	expected := vtl.from_array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3])!
	assert res.array_equal(expected)
}

fn test_eye_different_shape() {
	res := vtl.eye[f64](2, 4, 0)
	expected := vtl.from_array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [2, 4])!
	assert res.array_equal(expected)
}

fn test_eye_offset() {
	res := vtl.eye[f64](3, 3, 1)
	expected := vtl.from_array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [3, 3])!
	assert res.array_equal(expected)
}

fn test_identity() {
	res := vtl.identity[f64](3)
	expected := vtl.from_array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3])!
	assert res.array_equal(expected)
}

fn test_zeros() {
	mut t := vtl.zeros[f64]([3])
	assert t.get([0]) == 0.0
	assert t.get([1]) == 0.0
	assert t.get([2]) == 0.0
}

fn test_zeros_like() {
	mut t := vtl.zeros[f64]([3])
	t.fill(1.0)
	assert t.size() == 3
	assert t.get([0]) == 1.0
	assert t.get([1]) == 1.0
	assert t.get([2]) == 1.0
	mut t2 := vtl.zeros_like(t)
	assert t2.size() == 3
	assert t2.get([0]) == 0.0
	assert t2.get([1]) == 0.0
	assert t2.get([2]) == 0.0
}

fn test_ones() {
	mut t := vtl.ones[f64]([3])
	assert t.get([0]) == 1.0
	assert t.get([1]) == 1.0
	assert t.get([2]) == 1.0
}

fn test_ones_like() {
	mut t := vtl.ones[f64]([3])
	t.fill(0.0)
	mut t2 := vtl.ones_like(t)
	assert t2.size() == 3
	assert t2.get([0]) == 1.0
	assert t2.get([1]) == 1.0
	assert t2.get([2]) == 1.0
}

fn test_full() {
	mut t := vtl.full[f64]([3], 3.0)
	assert t.get([0]) == 3.0
	assert t.get([1]) == 3.0
	assert t.get([2]) == 3.0
}

fn test_full_like() {
	mut t := vtl.full[f64]([3], 3.0)
	mut t2 := vtl.full_like(t, 4.0)
	assert t2.size() == 3
	assert t2.get([0]) == 4.0
	assert t2.get([1]) == 4.0
	assert t2.get([2]) == 4.0
}

fn test_range() {
	t := vtl.range[f64](-2, 10)
	expected := vtl.from_array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [
		12,
	])!
	assert t.array_equal(expected)
}

fn test_seq() {
	t := vtl.seq[f64](10)
	expected := vtl.from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [
		10,
	])!
	assert t.array_equal(expected)
}

fn test_from_1d() {
	t := vtl.from_1d[f64]([1.0, 2.0, 3.0])!
	expected := vtl.from_array([1.0, 2.0, 3.0], [3])!
	assert t.array_equal(expected)
}

// Regression for #41: shape slice must be owned (Windows heap corruption if aliased).
fn test_from_array_shape_not_aliased() {
	mut sh := [4]
	t := vtl.from_array([f32(1), 2, 3, 4], sh)!
	sh[0] = 99
	assert t.shape[0] == 4
	assert t.get_nth(3) == 4
}

fn test_from_2d() {
	t := vtl.from_2d[f64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!
	expected := vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])!
	assert t.array_equal(expected)
}
