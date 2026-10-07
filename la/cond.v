module la

import math
import vtl

// cond computes a matrix condition number for each trailing square matrix.
// Supported orders match NumPy's real matrix orders: 2, -2, 1, -1, inf,
// -inf, and fro. A single matrix returns a one-element tensor.
pub struct CondOptions {
pub:
	ord string = '2'
}

pub fn cond[T](input &vtl.Tensor[T], options CondOptions) !&vtl.Tensor[f64] {
	order := match options.ord {
		'I' { 'inf' }
		'-I' { '-inf' }
		else { options.ord }
	}
	if order !in ['2', '-2', '1', '-1', 'inf', '-inf', 'fro'] {
		return error('cond: unsupported order `${options.ord}`')
	}
	if input.rank() < 2 {
		return error('cond requires input with rank at least 2')
	}
	rows := input.shape[input.rank() - 2]
	columns := input.shape[input.rank() - 1]
	if rows != columns {
		return error('cond requires square matrices, got ${rows}x${columns}')
	}
	batch_shape := input.shape[..input.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	output_shape := if batch_shape.len == 0 { [1] } else { batch_shape.clone() }
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut input_index := []int{len: input.rank()}
	for batch in 0 .. batch_count {
		decode_matrix_batch(batch, batch_shape, mut input_index)
		mut matrix := []f64{len: rows * columns}
		for row in 0 .. rows {
			input_index[input.rank() - 2] = row
			for column in 0 .. columns {
				input_index[input.rank() - 1] = column
				value := f64(input.get[T](input_index))
				if math.is_nan(value) || math.is_inf(value, 0) {
					return error('cond requires finite input values')
				}
				matrix[row * columns + column] = value
			}
		}
		mut condition := 0.0
		if order in ['2', '-2'] {
			values := matrix_singular_values(matrix, rows, columns)!
			if values.len > 0 {
				if order == '2' {
					condition = if values[values.len - 1] == 0 {
						math.inf(1)
					} else {
						values[0] / values[values.len - 1]
					}
				} else {
					condition = if values[0] == 0 { 0 } else { values[values.len - 1] / values[0] }
				}
			}
		} else {
			matrix_norm := matrix_norm_single(matrix, rows, columns, order)!
			inverse := invert_square_matrix(matrix, rows)!
			inverse_norm := matrix_norm_single(inverse, rows, columns, order)!
			condition = matrix_norm * inverse_norm
		}
		result.set_nth(batch, condition)
	}
	return result
}
