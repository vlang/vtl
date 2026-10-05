module main

import vtl
import math

fn test_meshgrid_xy_coordinates() {
	x := vtl.from_1d([1, 2, 3])!
	y := vtl.from_1d([10, 20])!
	x_grid, y_grid := vtl.meshgrid(x, y)!
	assert x_grid.shape == [2, 3]
	assert y_grid.shape == [2, 3]
	assert x_grid.array_equal[int](vtl.from_2d[int]([[1, 2, 3], [1, 2, 3]])!)
	assert y_grid.array_equal[int](vtl.from_2d[int]([[10, 10, 10], [20, 20, 20]])!)
}

fn test_meshgrid_requires_vectors() {
	x := vtl.from_2d([[1, 2]])!
	y := vtl.from_1d([1, 2])!
	if _, _ := vtl.meshgrid(x, y) {
		assert false, 'meshgrid must reject inputs above rank one'
	} else {
		assert true
	}
}

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

fn test_arange_float_step() ! {
	t := vtl.arange[f64](0.0, 1.0, 0.2)!
	assert t.size() == 5
	assert t.get_nth(0) == 0.0
	assert t.get_nth(1) == 0.2
	assert t.get_nth(2) == 0.4
	assert t.get_nth(3) == 0.6000000000000001
	assert t.get_nth(4) == 0.8
}

fn test_arange_integer_and_descending_steps() ! {
	ascending := vtl.arange[int](2, 8, 2)!
	assert ascending.to_array() == [2, 4, 6]

	descending := vtl.arange[int](5, 0, -2)!
	assert descending.to_array() == [5, 3, 1]

	empty := vtl.arange[int](0, 5, -1)!
	assert empty.size() == 0
}

fn test_arange_rejects_zero_and_non_finite_steps() {
	for step in [0.0, math.inf(1), math.nan()] {
		_ := vtl.arange[f64](0.0, 1.0, step) or {
			assert err.msg().contains('non-zero step')
			continue
		}
		assert false, 'expected arange to reject step ${step}'
	}
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
