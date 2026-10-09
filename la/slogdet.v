module la

import math
import math.complex as vcomplex
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

// slogdet_complex returns the unit-complex determinant phase and the natural
// logarithm of its magnitude for each complex128 square matrix. Leading batch
// dimensions are preserved. Singular matrices return a zero phase and -Inf.
pub fn slogdet_complex(input &vtl.Tensor[vcomplex.Complex]) !(&vtl.Tensor[vcomplex.Complex], &vtl.Tensor[f64]) {
	if input.rank() < 2 {
		return error('slogdet_complex requires input with rank at least 2')
	}
	n := input.shape[input.rank() - 1]
	if input.shape[input.rank() - 2] != n {
		return error('slogdet_complex requires square matrices')
	}
	batch_shape := input.shape[..input.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	output_shape := if batch_shape.len == 0 { [1] } else { batch_shape.clone() }
	mut phases := vtl.empty[vcomplex.Complex](output_shape, memory: .row_major)
	mut logabsdets := vtl.empty[f64](output_shape, memory: .row_major)
	mut index := []int{len: input.rank()}
	for batch in 0 .. batch_count {
		decode_matrix_batch(batch, batch_shape, mut index)
		mut matrix := []vcomplex.Complex{len: n * n}
		for row in 0 .. n {
			index[input.rank() - 2] = row
			for column in 0 .. n {
				index[input.rank() - 1] = column
				value := input.get[vcomplex.Complex](index)
				if !math.is_finite(value.re) || !math.is_finite(value.im) {
					return error('slogdet_complex requires finite input values')
				}
				matrix[row * n + column] = value
			}
		}
		mut phase := vcomplex.Complex{ re: 1 }
		mut logabsdet := 0.0
		mut singular := false
		for pivot_column in 0 .. n {
			mut pivot_row := pivot_column
			mut pivot_magnitude := complex_magnitude(matrix[pivot_column * n + pivot_column])
			for row in pivot_column + 1 .. n {
				magnitude := complex_magnitude(matrix[row * n + pivot_column])
				if magnitude > pivot_magnitude {
					pivot_row = row
					pivot_magnitude = magnitude
				}
			}
			if pivot_magnitude == 0 {
				singular = true
				break
			}
			if pivot_row != pivot_column {
				for column in 0 .. n {
					top := pivot_column * n + column
					bottom := pivot_row * n + column
					matrix[top], matrix[bottom] = matrix[bottom], matrix[top]
				}
				phase = vcomplex.Complex{ re: -phase.re, im: -phase.im }
			}
			pivot := matrix[pivot_column * n + pivot_column]
			pivot_magnitude = complex_magnitude(pivot)
			phase = phase.multiply(vcomplex.Complex{ re: pivot.re / pivot_magnitude, im: pivot.im / pivot_magnitude })
			logabsdet += math.log(pivot_magnitude)
			for row in pivot_column + 1 .. n {
				row_offset := row * n
				factor := matrix[row_offset + pivot_column] / pivot
				matrix[row_offset + pivot_column] = vcomplex.Complex{}
				for column in pivot_column + 1 .. n {
					position := row_offset + column
					matrix[position] = matrix[position].subtract(factor.multiply(matrix[pivot_column * n + column]))
				}
			}
		}
		if singular {
			phase = vcomplex.Complex{}
			logabsdet = math.inf(-1)
		}
		phases.set_nth(batch, phase)
		logabsdets.set_nth(batch, logabsdet)
	}
	return phases, logabsdets
}
