module core

import math
import vtl

// The contiguous fast paths in math.v must produce exactly the same values
// as the generic map_values path: f64 elements go straight to libm, f32
// elements are evaluated in f64 and rounded once, and non-contiguous
// tensors keep the generic path.

fn test_unary_fast_path_f64_matches_libm() ! {
	t := vtl.from_array([0.0, 1.0, -2.5, 10.0], [4])!
	exps := t.exp()
	sins := t.sin()
	logs := vtl.from_array([1.0, 2.718281828459045, 10.0, 100.0], [4])!.log()
	for i in 0 .. 4 {
		x := t.get_nth(i)
		assert exps.get_nth(i) == math.exp(x), 'exp mismatch at ${i}'
		assert sins.get_nth(i) == math.sin(x), 'sin mismatch at ${i}'
	}
	assert logs.get_nth(0) == 0.0
	assert math.abs(logs.get_nth(2) - math.log(10.0)) < 1e-15
}

fn test_unary_fast_path_f32_matches_f64_evaluation() ! {
	t := vtl.from_array([f32(0.0), f32(1.0), f32(-2.5), f32(10.0)], [4])!
	exps := t.exp()
	sins := t.sin()
	for i in 0 .. 4 {
		x := t.get_nth(i)
		assert exps.get_nth(i) == f32(math.exp(f64(x))), 'f32 exp mismatch at ${i}'
		assert sins.get_nth(i) == f32(math.sin(f64(x))), 'f32 sin mismatch at ${i}'
	}
}

fn test_unary_fast_path_non_contiguous_tensor() ! {
	base := vtl.from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3])!
	transposed := base.transpose([1, 0])!
	exps := transposed.exp()
	assert exps.shape == [3, 2]
	for i in 0 .. 3 {
		for j in 0 .. 2 {
			want := math.exp(base.get([j, i]))
			got := exps.get([i, j])
			assert got == want, 'non-contiguous exp mismatch at ${i},${j}: ${got} vs ${want}'
		}
	}
}
