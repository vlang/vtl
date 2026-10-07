module la

import math
import vtl

fn assert_svd_reconstruct(input &vtl.Tensor[f64], u &vtl.Tensor[f64], s &vtl.Tensor[f64], vt &vtl.Tensor[f64], tol f64) {
	rows := input.shape[input.rank() - 2]
	columns := input.shape[input.rank() - 1]
	k := if rows < columns { rows } else { columns }
	u_columns := u.shape[u.rank() - 1]
	vt_rows := vt.shape[vt.rank() - 2]
	for row in 0 .. rows {
		for column in 0 .. columns {
			mut sum := 0.0
			for singular in 0 .. k {
				sum += u.get([row, singular]) * s.get_nth(singular) * vt.get([
					singular,
					column,
				])
			}
			assert math.abs(sum - input.get([row, column])) < tol
		}
	}
	assert u_columns >= k
	assert vt_rows >= k
}

fn test_svd_reconstructs_tall_and_wide_matrices() ! {
	tall := vtl.from_2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])!
	u, s, vt := svd(tall, full_matrices: false)!
	assert u.shape == [3, 2]
	assert s.shape == [2]
	assert vt.shape == [2, 2]
	assert_svd_reconstruct(tall, u, s, vt, 1e-10)

	wide := vtl.from_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!
	u2, s2, vt2 := svd(wide, full_matrices: false)!
	assert u2.shape == [2, 2]
	assert s2.shape == [2]
	assert vt2.shape == [2, 3]
	assert_svd_reconstruct(wide, u2, s2, vt2, 1e-10)
	u3, s3, vt3 := svd(wide)!
	assert u3.shape == [2, 2]
	assert s3.shape == [2]
	assert vt3.shape == [3, 3]
	assert_svd_reconstruct(wide, u3, s3, vt3, 1e-10)
}

fn test_svd_full_shapes_and_orthogonality() ! {
	input := vtl.from_2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])!
	u, s, vt := svd(input)!
	assert u.shape == [3, 3]
	assert s.shape == [2]
	assert vt.shape == [2, 2]
	assert_svd_reconstruct(input, u, s, vt, 1e-10)
	for i in 0 .. 3 {
		for j in 0 .. 3 {
			mut dot := 0.0
			for row in 0 .. 3 {
				dot += u.get([row, i]) * u.get([row, j])
			}
			assert math.abs(dot - if i == j { 1.0 } else { 0.0 }) < 1e-10
		}
	}
}

fn test_svd_batched_rank_deficient_and_zero_inputs() ! {
	batch := vtl.from_array([1.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0], [2, 2, 2])!
	u, s, vt := svd(batch, full_matrices: false)!
	assert u.shape == [2, 2, 2]
	assert s.shape == [2, 2]
	assert vt.shape == [2, 2, 2]
	assert math.abs(s.get_nth(0) - 5.0) < 1e-10
	assert s.get_nth(1) < 1e-10
	assert math.abs(s.get_nth(2)) == 0
	assert math.abs(s.get_nth(3)) == 0
	mut u_dot := 0.0
	for row in 0 .. 2 {
		u_dot += u.get([0, row, 0]) * u.get([0, row, 1])
	}
	assert math.abs(u_dot) < 1e-10
}

fn test_svd_empty_and_non_finite_inputs() ! {
	empty := vtl.empty[f64]([3, 0])
	u, s, vt := svd(empty)!
	assert u.shape == [3, 3]
	assert s.shape == [0]
	assert vt.shape == [0, 0]
	assert u.get([0, 0]) == 1.0
	assert u.get([1, 1]) == 1.0
	assert u.get([2, 2]) == 1.0
	wide_empty := vtl.empty[f64]([0, 3])
	_, _, wide_vt := svd(wide_empty)!
	assert wide_vt.shape == [3, 3]
	assert wide_vt.get([0, 0]) == 1.0
	assert wide_vt.get([1, 1]) == 1.0
	assert wide_vt.get([2, 2]) == 1.0
	bad := vtl.from_2d([[f64(math.inf(1))]])!
	if _, _, _ := svd(bad) {
		assert false, 'svd must reject non-finite values'
	} else {
		assert true
	}
}
