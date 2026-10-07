module la

import math
import vtl

// MatrixNormOptions selects a NumPy-compatible matrix norm and output shape.
@[params]
pub struct MatrixNormOptions {
pub:
	ord      string = 'fro'
	keepdims bool
}

// matrix_norm computes a matrix norm for each trailing matrix in an N-D
// tensor. Supported orders are 'fro', 'nuc', '1', '-1', '2', '-2', 'inf',
// and '-inf'. The default Frobenius result is returned as a one-element
// tensor for a single matrix; keepdims retains the final two unit dimensions.
pub fn matrix_norm[T](input &vtl.Tensor[T], options MatrixNormOptions) !&vtl.Tensor[f64] {
	if input.rank() < 2 {
		return error('matrix_norm requires input with rank at least 2')
	}
	ord := match options.ord {
		'F' { 'fro' }
		'I' { 'inf' }
		'-I' { '-inf' }
		else { options.ord }
	}
	if ord !in ['fro', 'nuc', '1', '-1', '2', '-2', 'inf', '-inf'] {
		return error('matrix_norm: unsupported order `${options.ord}`')
	}
	rows := input.shape[input.rank() - 2]
	columns := input.shape[input.rank() - 1]
	batch_shape := input.shape[..input.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	mut output_shape := batch_shape.clone()
	if options.keepdims {
		output_shape << 1
		output_shape << 1
	} else if output_shape.len == 0 {
		output_shape << 1
	}
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut input_index := []int{len: input.rank()}
	for batch in 0 .. batch_count {
		decode_matrix_batch(batch, batch_shape, mut input_index)
		mut matrix_data := []f64{len: rows * columns}
		for row in 0 .. rows {
			input_index[input.rank() - 2] = row
			for column in 0 .. columns {
				input_index[input.rank() - 1] = column
				matrix_data[column * rows + row] = f64(input.get[T](input_index))
			}
		}
		value := matrix_norm_single(matrix_data, rows, columns, ord) or {
			return error('matrix_norm: ${err}')
		}
		result.set_nth(batch, value)
	}
	return result
}

fn matrix_norm_single(data []f64, rows int, columns int, ord string) !f64 {
	if ord == 'fro' {
		return vector_norm2_f64(data)
	}
	if rows == 0 || columns == 0 {
		if ord in ['nuc', '2'] {
			return 0
		}
		if ord == '-2' || (ord in ['1', '-1'] && columns == 0)
			|| (ord in ['inf', '-inf'] && rows == 0) {
			return error('order `${ord}` is undefined for an empty matrix reduction')
		}
		return 0
	}
	if ord in ['1', '-1', 'inf', '-inf'] {
		axis_count := if ord in ['1', '-1'] { columns } else { rows }
		mut best := if ord in ['-1', '-inf'] { math.inf(1) } else { 0.0 }
		mut has_nan := false
		for axis in 0 .. axis_count {
			mut sum := 0.0
			inner_count := if ord in ['1', '-1'] { rows } else { columns }
			for inner in 0 .. inner_count {
				index := if ord in ['1', '-1'] { axis * rows + inner } else { inner * rows + axis }
				value := math.abs(data[index])
				if math.is_nan(value) {
					has_nan = true
				}
				sum += value
			}
			if (ord in ['-1', '-inf'] && sum < best) || (ord in ['1', 'inf'] && sum > best) {
				best = sum
			}
		}
		return if has_nan { math.nan() } else { best }
	}
	singular_values := matrix_singular_values(data, rows, columns)!
	if ord == '2' {
		return singular_values[0]
	}
	if ord == '-2' {
		return singular_values[singular_values.len - 1]
	}
	mut total := 0.0
	for singular_value in singular_values {
		total += singular_value
	}
	return total
}

fn matrix_singular_values(data []f64, rows int, columns int) ![]f64 {
	mut max_magnitude := 0.0
	for value in data {
		if math.is_nan(value) || math.is_inf(value, 0) {
			return error('spectral matrix norms require finite input values')
		}
		max_magnitude = math.max(max_magnitude, math.abs(value))
	}
	count := if rows < columns { rows } else { columns }
	if max_magnitude == 0 {
		return []f64{len: count}
	}
	// One-sided Jacobi works on the matrix directly, avoiding the loss of
	// precision caused by forming AᵀA. Transpose wide matrices to keep the
	// working column count no larger than the row count.
	work_rows := if rows < columns { columns } else { rows }
	work_columns := count
	mut work := []f64{len: work_rows * work_columns}
	if rows >= columns {
		for i, value in data {
			work[i] = value / max_magnitude
		}
	} else {
		for row in 0 .. rows {
			for column in 0 .. columns {
				work[row * work_rows + column] = data[column * rows + row] / max_magnitude
			}
		}
	}
	mut converged := false
	for _ in 0 .. 100 {
		mut rotated := false
		for p in 0 .. work_columns {
			for q in p + 1 .. work_columns {
				mut alpha := 0.0
				mut beta := 0.0
				mut gamma := 0.0
				for row in 0 .. work_rows {
					x := work[p * work_rows + row]
					y := work[q * work_rows + row]
					alpha += x * x
					beta += y * y
					gamma += x * y
				}
				if gamma == 0 || math.abs(gamma) <= 1e-14 * math.sqrt(alpha * beta) {
					continue
				}
				zeta := (beta - alpha) / (2 * gamma)
				mut tangent := 1.0 / (math.abs(zeta) + math.sqrt(1 + zeta * zeta))
				if zeta < 0 {
					tangent = -tangent
				}
				cosine := 1 / math.sqrt(1 + tangent * tangent)
				sine := cosine * tangent
				for row in 0 .. work_rows {
					p_index := p * work_rows + row
					q_index := q * work_rows + row
					x := work[p_index]
					y := work[q_index]
					work[p_index] = cosine * x - sine * y
					work[q_index] = sine * x + cosine * y
				}
				rotated = true
			}
		}
		if !rotated {
			converged = true
			break
		}
	}
	if !converged {
		return error('one-sided Jacobi SVD did not converge after 100 sweeps')
	}
	mut singular_values := []f64{len: work_columns}
	for column in 0 .. work_columns {
		mut sum_squares := 0.0
		for row in 0 .. work_rows {
			value := work[column * work_rows + row]
			sum_squares += value * value
		}
		singular_values[column] = math.sqrt(sum_squares) * max_magnitude
	}
	for i in 0 .. singular_values.len {
		for j in i + 1 .. singular_values.len {
			if singular_values[j] > singular_values[i] {
				singular_values[i], singular_values[j] = singular_values[j], singular_values[i]
			}
		}
	}
	return singular_values
}

fn decode_matrix_batch(batch int, batch_shape []int, mut index []int) {
	mut remainder := batch
	for axis := batch_shape.len - 1; axis >= 0; axis-- {
		index[axis] = remainder % batch_shape[axis]
		remainder /= batch_shape[axis]
	}
}
