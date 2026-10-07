module la

import math
import vtl

// SVDOptions selects whether svd returns full orthogonal matrices or the
// reduced factors. The default matches NumPy's full_matrices=true behavior.
@[params]
pub struct SVDOptions {
pub:
	full_matrices bool = true
}

// svd computes the singular value decomposition of every trailing real
// matrix, returning U, descending singular values, and V transpose. Results
// use f64. Reduced factors have shapes [..., M, K], [..., K], [..., K, N],
// where K=min(M,N); full factors use [..., M, M] and [..., N, N].
pub fn svd[T](input &vtl.Tensor[T], options SVDOptions) !(&vtl.Tensor[f64], &vtl.Tensor[f64], &vtl.Tensor[f64]) {
	if input.rank() < 2 {
		return error('svd requires input with rank at least 2')
	}
	rows := input.shape[input.rank() - 2]
	columns := input.shape[input.rank() - 1]
	k := if rows < columns { rows } else { columns }
	batch_shape := input.shape[..input.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	mut u_shape := batch_shape.clone()
	mut s_shape := batch_shape.clone()
	mut vt_shape := batch_shape.clone()
	u_shape << rows
	u_shape << if options.full_matrices { rows } else { k }
	s_shape << k
	vt_shape << if options.full_matrices { columns } else { k }
	vt_shape << columns
	mut u := vtl.empty[f64](u_shape, memory: .row_major)
	mut singular_values := vtl.empty[f64](s_shape, memory: .row_major)
	mut vt_result := vtl.empty[f64](vt_shape, memory: .row_major)
	if rows == 0 || columns == 0 {
		if options.full_matrices {
			for batch in 0 .. batch_count {
				for diagonal in 0 .. rows {
					u.set_nth(batch * rows * rows + diagonal * rows + diagonal, 1)
				}
				for diagonal in 0 .. columns {
					vt_result.set_nth(batch * columns * columns + diagonal * columns + diagonal, 1)
				}
			}
		}
		return u, singular_values, vt_result
	}
	mut input_index := []int{len: input.rank()}
	for batch in 0 .. batch_count {
		decode_matrix_batch(batch, batch_shape, mut input_index)
		mut matrix := []f64{len: rows * columns}
		for row in 0 .. rows {
			input_index[input.rank() - 2] = row
			for column in 0 .. columns {
				input_index[input.rank() - 1] = column
				matrix[row * columns + column] = f64(input.get[T](input_index))
			}
		}
		factors := matrix_svd_f64(matrix, rows, columns, options.full_matrices) or {
			return error('svd: ${err}')
		}
		for i, value in factors.values {
			singular_values.set_nth(batch * k + i, value)
		}
		for i, value in factors.u {
			u.set_nth(batch * rows * u_shape[u_shape.len - 1] + i, value)
		}
		for i, value in factors.vt {
			vt_result.set_nth(batch * vt_shape[vt_shape.len - 2] * columns + i, value)
		}
	}
	return u, singular_values, vt_result
}

struct SvdFactors {
	values []f64
	u      []f64 // row-major rows by u_columns
	vt     []f64 // row-major vt_rows by columns
}

fn matrix_svd_tall(data []f64, rows int, columns int, full bool) !SvdFactors {
	mut scale := 0.0
	for value in data {
		if math.is_nan(value) || math.is_inf(value, 0) {
			return error('input values must be finite')
		}
		scale = math.max(scale, math.abs(value))
	}
	u_columns := if full { rows } else { columns }
	if scale == 0 {
		u := orthogonal_completion([]f64{len: rows * columns}, rows, columns, u_columns)
		vt := identity_f64(columns)
		return SvdFactors{ values: []f64{len: columns}, u: u, vt: vt }
	}
	mut work := []f64{len: rows * columns}
	for row in 0 .. rows {
		for column in 0 .. columns {
			work[column * rows + row] = data[row * columns + column] / scale
		}
	}
	mut right := identity_f64(columns)
	mut converged := false
	for _ in 0 .. 100 {
		mut rotated := false
		for p in 0 .. columns {
			for q in p + 1 .. columns {
				mut alpha := 0.0
				mut beta := 0.0
				mut gamma := 0.0
				for row in 0 .. rows {
					x := work[p * rows + row]
					y := work[q * rows + row]
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
				for row in 0 .. rows {
					p_index := p * rows + row
					q_index := q * rows + row
					x := work[p_index]
					y := work[q_index]
					work[p_index] = cosine * x - sine * y
					work[q_index] = sine * x + cosine * y
				}
				for row in 0 .. columns {
					p_index := p * columns + row
					q_index := q * columns + row
					x := right[p_index]
					y := right[q_index]
					right[p_index] = cosine * x - sine * y
					right[q_index] = sine * x + cosine * y
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
	mut values := []f64{len: columns}
	mut order := []int{len: columns}
	for column in 0 .. columns {
		mut norm_squared := 0.0
		for row in 0 .. rows {
			value := work[column * rows + row]
			norm_squared += value * value
		}
		values[column] = math.sqrt(norm_squared) * scale
		order[column] = column
	}
	for i in 0 .. columns {
		for j in i + 1 .. columns {
			if values[order[j]] > values[order[i]] {
				order[i], order[j] = order[j], order[i]
			}
		}
	}
	mut sorted_values := []f64{len: columns}
	mut sorted_u := []f64{len: rows * columns}
	mut sorted_vt := []f64{len: columns * columns}
	for output_column, source_column in order {
		sigma := values[source_column]
		sorted_values[output_column] = sigma
		for row in 0 .. rows {
			sorted_u[row * columns + output_column] = if sigma == 0 {
				0
			} else {
				work[source_column * rows + row] / (sigma / scale)
			}
		}
		for column in 0 .. columns {
			sorted_vt[output_column * columns + column] = right[source_column * columns + column]
		}
	}
	sorted_u = orthogonal_completion(sorted_u, rows, columns, u_columns)
	return SvdFactors{ values: sorted_values, u: sorted_u, vt: sorted_vt }
}

fn orthogonal_completion(input []f64, rows int, input_columns int, output_columns int) []f64 {
	mut output := []f64{len: rows * output_columns}
	for row in 0 .. rows {
		for column in 0 .. input_columns {
			output[row * output_columns + column] = input[row * input_columns + column]
		}
	}
	for column in 0 .. output_columns {
		if column < input_columns {
			for previous in 0 .. column {
				mut projection := 0.0
				for row in 0 .. rows {
					projection += output[row * output_columns + column] * output[row * output_columns + previous]
				}
				for row in 0 .. rows {
					output[row * output_columns + column] -= projection * output[row * output_columns + previous]
				}
			}
		}
		mut norm_squared := 0.0
		for row in 0 .. rows {
			value := output[row * output_columns + column]
			norm_squared += value * value
		}
		if norm_squared < 1e-28 {
			mut found := false
			for basis in 0 .. rows {
				for row in 0 .. rows {
					output[row * output_columns + column] = if row == basis { 1 } else { 0 }
				}
				for previous in 0 .. column {
					mut projection := 0.0
					for row in 0 .. rows {
						projection += output[row * output_columns + column] * output[row * output_columns + previous]
					}
					for row in 0 .. rows {
						output[row * output_columns + column] -= projection * output[row * output_columns + previous]
					}
				}
				norm_squared = 0
				for row in 0 .. rows {
					value := output[row * output_columns + column]
					norm_squared += value * value
				}
				if norm_squared > 1e-28 {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		column_norm := math.sqrt(norm_squared)
		for row in 0 .. rows {
			output[row * output_columns + column] /= column_norm
		}
	}
	return output
}

fn identity_f64(size int) []f64 {
	mut result := []f64{len: size * size}
	for i in 0 .. size {
		result[i * size + i] = 1
	}
	return result
}
