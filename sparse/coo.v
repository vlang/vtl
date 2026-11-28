module sparse

import vtl

struct CooKey {
	row int
	col int
	idx int
}

// CooTensor is a two-dimensional sparse tensor in coordinate (COO) format.
// Rows, cols and values are parallel arrays. Duplicate coordinates and
// explicit zeros are permitted; to_dense sums values in insertion order.
pub struct CooTensor[T] {
pub:
	shape  []int
	rows   []int
	cols   []int
	values []T
}

// from_coo builds a COO tensor and validates the shape and every coordinate.
pub fn from_coo[T](shape []int, rows []int, cols []int, values []T) !CooTensor[T] {
	if shape.len != 2 {
		return error('COO tensors require a 2-D shape')
	}
	if shape[0] < 0 || shape[1] < 0 {
		return error('COO shape dimensions must be non-negative')
	}
	if rows.len != cols.len || rows.len != values.len {
		return error('COO rows, cols and values must have the same length')
	}
	for i in 0 .. rows.len {
		if rows[i] < 0 || rows[i] >= shape[0] {
			return error('COO row index ${rows[i]} is outside shape[0]')
		}
		if cols[i] < 0 || cols[i] >= shape[1] {
			return error('COO column index ${cols[i]} is outside shape[1]')
		}
	}
	return CooTensor[T]{
		shape:  shape.clone()
		rows:   rows.clone()
		cols:   cols.clone()
		values: values.clone()
	}
}

// from_dense extracts the non-zero coordinates of a 2-D dense tensor.
pub fn from_dense[T](dense &vtl.Tensor[T]) !CooTensor[T] {
	if dense.rank() != 2 {
		return error('from_dense requires a 2-D tensor')
	}
	mut rows := []int{}
	mut cols := []int{}
	mut values := []T{}
	for i in 0 .. dense.shape[0] {
		for j in 0 .. dense.shape[1] {
			value := dense.get[T]([i, j])
			if value != T(0) {
				rows << i
				cols << j
				values << value
			}
		}
	}
	return from_coo[T](dense.shape, rows, cols, values)
}

// nnz reports the number of stored coordinates, including duplicates and
// explicit zeros.
pub fn (t &CooTensor[T]) nnz() int {
	return t.values.len
}

// to_dense expands the COO tensor into a dense row-major tensor. Values at
// the same coordinate are summed in insertion order.
pub fn (t &CooTensor[T]) to_dense() !&vtl.Tensor[T] {
	if t.shape[0] != 0 && t.shape[1] > max_int / t.shape[0] {
		return error('COO shape overflows dense size')
	}
	mut data := []T{len: t.shape[0] * t.shape[1]}
	for i in 0 .. t.values.len {
		data[t.rows[i] * t.shape[1] + t.cols[i]] += t.values[i]
	}
	return vtl.from_array(data, t.shape.clone())
}

// coalesce returns a canonical COO tensor with coordinates sorted by row and
// column and duplicates summed. Explicit zeros are retained so that the
// stored count stays predictable.
pub fn (t &CooTensor[T]) coalesce() !CooTensor[T] {
	mut keys := []CooKey{len: t.values.len}
	for i in 0 .. keys.len {
		keys[i] = CooKey{t.rows[i], t.cols[i], i}
	}
	keys.sort_with_compare(fn (a &CooKey, b &CooKey) int {
		if a.row < b.row {
			return -1
		}
		if a.row > b.row {
			return 1
		}
		if a.col < b.col {
			return -1
		}
		if a.col > b.col {
			return 1
		}
		return 0
	})
	mut rows := []int{}
	mut cols := []int{}
	mut values := []T{}
	for key in keys {
		r := key.row
		c := key.col
		idx := key.idx
		if rows.len > 0 && rows.last() == r && cols.last() == c {
			values[values.len - 1] += t.values[idx]
		} else {
			rows << r
			cols << c
			values << t.values[idx]
		}
	}
	return from_coo[T](t.shape, rows, cols, values)
}
