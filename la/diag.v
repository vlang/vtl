module la

import vtl

// diag constructs a diagonal matrix from a vector or extracts a diagonal from
// a matrix. A positive offset moves above the main diagonal; a negative offset
// moves below it, matching NumPy's diag convention. The result is a copy.
pub fn diag[T](input &vtl.Tensor[T], offset int) !&vtl.Tensor[T] {
	if input.rank() == 1 {
		length := input.size + if offset < 0 { -offset } else { offset }
		mut output := []T{len: length * length}
		start_row := if offset < 0 { -offset } else { 0 }
		start_column := if offset > 0 { offset } else { 0 }
		for index in 0 .. input.size {
			output[(start_row + index) * length + start_column + index] = input.get_nth(index)
		}
		return vtl.from_array[T](output, [length, length])
	}
	if input.rank() != 2 {
		return error('diag expects a one-dimensional vector or a two-dimensional matrix')
	}
	row_start := if offset < 0 { -offset } else { 0 }
	column_start := if offset > 0 { offset } else { 0 }
	rows_remaining := input.shape[0] - row_start
	columns_remaining := input.shape[1] - column_start
	length := if rows_remaining < columns_remaining { rows_remaining } else { columns_remaining }
	if length <= 0 {
		return vtl.from_array[T]([], [0])
	}
	mut diagonal := []T{len: length}
	for index in 0 .. length {
		diagonal[index] = input.get([row_start + index, column_start + index])
	}
	return vtl.from_array[T](diagonal, [length])
}
