module la

import math
import vtl

// matrix_power raises each trailing square matrix to an integer power using
// exponentiation by squaring. Negative powers use a partial-pivot inverse.
// Results are f64 tensors with the same shape as the input.
pub fn matrix_power[T](input &vtl.Tensor[T], exponent int) !&vtl.Tensor[f64] {
	if input.rank() < 2 {
		return error('matrix_power requires input with rank at least 2')
	}
	rows := input.shape[input.rank() - 2]
	columns := input.shape[input.rank() - 1]
	if rows != columns {
		return error('matrix_power requires square matrices, got ${rows}x${columns}')
	}
	batch_shape := input.shape[..input.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	mut result := vtl.empty[f64](input.shape, memory: .row_major)
	mut input_index := []int{len: input.rank()}
	for batch in 0 .. batch_count {
		decode_matrix_batch(batch, batch_shape, mut input_index)
		mut base := []f64{len: rows * columns}
		for row in 0 .. rows {
			input_index[input.rank() - 2] = row
			for column in 0 .. columns {
				input_index[input.rank() - 1] = column
				base[row * columns + column] = f64(input.get[T](input_index))
			}
		}
		if exponent < 0 {
			base = invert_square_matrix(base, rows)!
		}
		mut power := u64(if exponent < 0 { -(exponent + 1) } else { exponent })
		if exponent < 0 {
			power++
		}
		mut powered := []f64{len: rows * columns}
		for i in 0 .. rows {
			powered[i * rows + i] = 1.0
		}
		for power > 0 {
			if power % 2 == 1 {
				powered = multiply_square_matrices(powered, base, rows)
			}
			power /= 2
			if power > 0 {
				base = multiply_square_matrices(base, base, rows)
			}
		}
		start := batch * rows * columns
		for i, value in powered {
			result.set_nth(start + i, value)
		}
	}
	return result
}

fn multiply_square_matrices(a []f64, b []f64, size int) []f64 {
	mut output := []f64{len: size * size}
	for row in 0 .. size {
		for inner in 0 .. size {
			value := a[row * size + inner]
			for column in 0 .. size {
				output[row * size + column] += value * b[inner * size + column]
			}
		}
	}
	return output
}

fn invert_square_matrix(input []f64, size int) ![]f64 {
	mut left := input.clone()
	mut right := []f64{len: size * size}
	for i in 0 .. size {
		right[i * size + i] = 1.0
	}
	for pivot_column in 0 .. size {
		mut pivot_row := pivot_column
		for row in pivot_column + 1 .. size {
			if math.abs(left[row * size + pivot_column]) > math.abs(left[pivot_row * size + pivot_column]) {
				pivot_row = row
			}
		}
		pivot := left[pivot_row * size + pivot_column]
		if pivot == 0 {
			return error('matrix_power cannot raise a singular matrix to a negative power')
		}
		if pivot_row != pivot_column {
			for column in 0 .. size {
				left_a := pivot_column * size + column
				left_b := pivot_row * size + column
				right_a := pivot_column * size + column
				right_b := pivot_row * size + column
				left[left_a], left[left_b] = left[left_b], left[left_a]
				right[right_a], right[right_b] = right[right_b], right[right_a]
			}
		}
		pivot_value := left[pivot_column * size + pivot_column]
		for column in 0 .. size {
			left[pivot_column * size + column] /= pivot_value
			right[pivot_column * size + column] /= pivot_value
		}
		for row in 0 .. size {
			if row == pivot_column {
				continue
			}
			factor := left[row * size + pivot_column]
			for column in 0 .. size {
				left[row * size + column] -= factor * left[pivot_column * size + column]
				right[row * size + column] -= factor * right[pivot_column * size + column]
			}
		}
	}
	return right
}
