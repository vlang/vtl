module sparse

import vtl
import vtl.sparse as sp

fn test_from_coo_validates_shape_and_indices() ! {
	_ = sp.from_coo[f64]([2, 3], [0, 1], [1, 2], [1.0, 2.0])!

	sp.from_coo[f64]([2], [0], [0], [1.0]) or {
		assert err.msg().len > 0
		return
	}
	assert false, 'non-2D shape should be rejected'
}

fn test_from_coo_rejects_mismatched_coordinate_arrays() ! {
	sp.from_coo[f64]([2, 3], [0, 2], [1], [1.0, 2.0]) or {
		assert err.msg().len > 0
		return
	}
	assert false, 'mismatched coordinate arrays should be rejected'
}

fn test_from_coo_rejects_out_of_range_indices() ! {
	sp.from_coo[f64]([2, 3], [0, 2], [1, 1], [1.0, 2.0]) or {
		assert err.msg().len > 0
		return
	}
	assert false, 'out-of-range row should be rejected'
}

fn test_from_coo_rejects_out_of_range_columns() ! {
	sp.from_coo[f64]([2, 3], [0, 1], [1, 3], [1.0, 2.0]) or {
		assert err.msg().len > 0
		return
	}
	assert false, 'out-of-range column should be rejected'
}

fn test_coo_dense_round_trip_sums_duplicates() ! {
	tensor := sp.from_coo[f64]([2, 3], [0, 0, 1], [1, 1, 2], [1.5, 2.5, -3.0])!
	assert tensor.nnz() == 3
	dense := tensor.to_dense()!
	assert dense.shape == [2, 3]
	assert dense.to_array() == [0.0, 4.0, 0.0, 0.0, 0.0, -3.0]
}

fn test_coo_from_dense_skips_explicit_zeros() ! {
	dense := vtl.from_array([f64(0), 7, 0, 0, 8, 0], [2, 3])!
	tensor := sp.from_dense(dense)!
	assert tensor.nnz() == 2
	assert tensor.rows == [0, 1]
	assert tensor.cols == [1, 1]
	assert tensor.values == [f64(7), 8]
	assert tensor.to_dense()!.to_array() == dense.to_array()
}

fn test_coo_coalesce_sorts_and_merges_duplicates() ! {
	tensor := sp.from_coo[f64]([3, 3], [2, 0, 2, 0], [0, 2, 0, 1], [3.0, 5.0, 4.0, 1.0])!
	merged := tensor.coalesce()!
	assert merged.nnz() == 3
	assert merged.rows == [0, 0, 2]
	assert merged.cols == [1, 2, 0]
	assert merged.values == [f64(1), 5, 7]
	assert merged.to_dense()!.to_array() == tensor.to_dense()!.to_array()
}

fn test_coo_handles_empty_and_zero_sized_dimensions() ! {
	empty := sp.from_coo[f64]([0, 3], [], [], [])!
	assert empty.nnz() == 0
	assert empty.to_dense()!.size == 0
	assert empty.coalesce()!.nnz() == 0

	rows := sp.from_coo[f64]([2, 0], [], [], [])!
	assert rows.to_dense()!.shape == [2, 0]
}
