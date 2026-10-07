module la

import math
import vtl

// slogdet returns each matrix's determinant sign and the natural logarithm
// of its absolute determinant. The two results have the leading batch shape;
// for a single matrix they contain one value. Input matrices must be finite
// and square, and may be stacked across leading dimensions.
pub fn slogdet[T](input &vtl.Tensor[T]) !(&vtl.Tensor[f64], &vtl.Tensor[f64]) {
	if input.rank() < 2 {
		return error('slogdet requires input with rank at least 2')
	}
	rows := input.shape[input.rank() - 2]
	columns := input.shape[input.rank() - 1]
	if rows != columns {
		return error('slogdet requires square matrices, got ${rows}x${columns}')
	}
	batch_shape := input.shape[..input.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	output_shape := if batch_shape.len == 0 { [1] } else { batch_shape.clone() }
	mut signs := vtl.empty[f64](output_shape, memory: .row_major)
	mut logabsdets := vtl.empty[f64](output_shape, memory: .row_major)
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
					return error('slogdet requires finite input values')
				}
				matrix[row * columns + column] = value
			}
		}
		mut sign := 1.0
		mut logabsdet := 0.0
		mut singular := false
		for pivot_column in 0 .. columns {
			mut pivot_row := pivot_column
			for row in pivot_column + 1 .. rows {
				if math.abs(matrix[row * columns + pivot_column]) > math.abs(matrix[pivot_row * columns + pivot_column]) {
					pivot_row = row
				}
			}
			pivot := matrix[pivot_row * columns + pivot_column]
			if pivot == 0 {
				singular = true
				break
			}
			if pivot_row != pivot_column {
				for column in pivot_column .. columns {
					left := pivot_column * columns + column
					right := pivot_row * columns + column
					matrix[left], matrix[right] = matrix[right], matrix[left]
				}
				sign = -sign
			}
			current_pivot := matrix[pivot_column * columns + pivot_column]
			if current_pivot < 0 {
				sign = -sign
			}
			logabsdet += math.log(math.abs(current_pivot))
			for row in pivot_column + 1 .. rows {
				factor := matrix[row * columns + pivot_column] / current_pivot
				for column in pivot_column + 1 .. columns {
					matrix[row * columns + column] -= factor * matrix[pivot_column * columns + column]
				}
			}
		}
		if singular {
			sign = 0
			logabsdet = math.inf(-1)
		}
		signs.set_nth(batch, sign)
		logabsdets.set_nth(batch, logabsdet)
	}
	return signs, logabsdets
}
