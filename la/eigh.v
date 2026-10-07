module la

import math
import vtl

// EighOptions selects the triangle used to construct each symmetric matrix.
// The default lower triangle matches NumPy's default UPLO.
pub struct EighOptions {
pub:
	uplo string = 'L'
}

// eigh computes ascending eigenvalues and corresponding eigenvectors for each
// trailing real symmetric matrix. Eigenvectors are stored in columns, and the
// results have shapes [..., N] and [..., N, N]. Non-finite inputs and matrices
// that do not converge within 100 Jacobi sweeps return errors.
pub fn eigh[T](input &vtl.Tensor[T], options EighOptions) !(&vtl.Tensor[f64], &vtl.Tensor[f64]) {
	return symmetric_eigen[T](input, options, true)
}

// eigvalsh returns only the ascending eigenvalues of trailing real symmetric
// matrices. The UPLO selection matches EighOptions and skips eigenvector work.
pub fn eigvalsh[T](input &vtl.Tensor[T], options EighOptions) !&vtl.Tensor[f64] {
	values, _ := symmetric_eigen[T](input, options, false)!
	return values
}

fn symmetric_eigen[T](input &vtl.Tensor[T], options EighOptions, compute_vectors bool) !(&vtl.Tensor[f64], &vtl.Tensor[f64]) {
	if input.rank() < 2 {
		return error('eigh requires input with rank at least 2')
	}
	rows := input.shape[input.rank() - 2]
	columns := input.shape[input.rank() - 1]
	if rows != columns {
		return error('eigh requires square matrices, got ${rows}x${columns}')
	}
	uplo := options.uplo.to_upper()
	if uplo !in ['L', 'U'] {
		return error('eigh: UPLO must be L or U')
	}
	batch_shape := input.shape[..input.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	mut eigenvalue_shape := batch_shape.clone()
	eigenvalue_shape << rows
	mut eigenvector_shape := batch_shape.clone()
	eigenvector_shape << rows
	eigenvector_shape << rows
	mut eigenvalues := vtl.empty[f64](eigenvalue_shape, memory: .row_major)
	mut eigenvectors := if compute_vectors {
		vtl.empty[f64](eigenvector_shape, memory: .row_major)
	} else {
		vtl.empty[f64]([0])
	}
	if rows == 0 {
		return eigenvalues, eigenvectors
	}
	mut input_index := []int{len: input.rank()}
	for batch in 0 .. batch_count {
		decode_matrix_batch(batch, batch_shape, mut input_index)
		mut matrix := [][]f64{len: rows, init: []f64{len: rows}}
		mut scale := 0.0
		for row in 0 .. rows {
			for column in 0 .. rows {
				selected_row := if uplo == 'L' {
					if row >= column { row } else { column }
				} else {
					if row <= column { row } else { column }
				}
				selected_column := if uplo == 'L' {
					if row >= column { column } else { row }
				} else {
					if row <= column { column } else { row }
				}
				input_index[input.rank() - 2] = selected_row
				input_index[input.rank() - 1] = selected_column
				value := f64(input.get[T](input_index))
				if math.is_nan(value) || math.is_inf(value, 0) {
					return error('eigh requires finite input values')
				}
				matrix[row][column] = value
				scale = math.max(scale, math.abs(value))
			}
		}
		if scale > 0 {
			for row in 0 .. rows {
				for column in 0 .. rows {
					matrix[row][column] /= scale
				}
			}
		}
		values := symmetric_jacobi(mut matrix, compute_vectors)!
		value_start := batch * rows
		for i, value in values {
			eigenvalues.set_nth(value_start + i, value * scale)
		}
		if compute_vectors {
			vector_start := batch * rows * rows
			for row in 0 .. rows {
				for column in 0 .. rows {
					eigenvectors.set_nth(vector_start + row * rows + column, matrix[row][column])
				}
			}
		}
	}
	return eigenvalues, eigenvectors
}

fn symmetric_jacobi(mut matrix [][]f64, compute_vectors bool) ![]f64 {
	n := matrix.len
	mut vectors := [][]f64{len: n, init: []f64{len: n}}
	if compute_vectors {
		for i in 0 .. n {
			vectors[i][i] = 1.0
		}
	}
	max_sweeps := 100
	epsilon := 1e-15
	for _ in 0 .. max_sweeps {
		mut max_off_diagonal := 0.0
		for p in 0 .. n {
			for q in p + 1 .. n {
				apq := matrix[p][q]
				max_off_diagonal = math.max(max_off_diagonal, math.abs(apq))
				if math.abs(apq) <= epsilon {
					continue
				}
				app := matrix[p][p]
				aqq := matrix[q][q]
				tau := (aqq - app) / (2.0 * apq)
				mut tangent := 1.0 / (math.abs(tau) + math.sqrt(1.0 + tau * tau))
				if tau < 0 {
					tangent = -tangent
				}
				cosine := 1.0 / math.sqrt(1.0 + tangent * tangent)
				sine := tangent * cosine
				for k in 0 .. n {
					if k == p || k == q {
						continue
					}
					akp := matrix[k][p]
					akq := matrix[k][q]
					matrix[k][p] = cosine * akp - sine * akq
					matrix[p][k] = matrix[k][p]
					matrix[k][q] = sine * akp + cosine * akq
					matrix[q][k] = matrix[k][q]
				}
				matrix[p][p] = app - tangent * apq
				matrix[q][q] = aqq + tangent * apq
				matrix[p][q] = 0.0
				matrix[q][p] = 0.0
				if compute_vectors {
					for k in 0 .. n {
						vkp := vectors[k][p]
						vkq := vectors[k][q]
						vectors[k][p] = cosine * vkp - sine * vkq
						vectors[k][q] = sine * vkp + cosine * vkq
					}
				}
			}
		}
		if max_off_diagonal <= epsilon {
			break
		}
	}
	mut remaining_off_diagonal := 0.0
	for row in 0 .. n {
		for column in row + 1 .. n {
			remaining_off_diagonal = math.max(remaining_off_diagonal, math.abs(matrix[row][column]))
		}
	}
	if remaining_off_diagonal > epsilon {
		return error('eigh Jacobi iteration did not converge within ${max_sweeps} sweeps')
	}
	mut indices := []int{len: n}
	mut values := []f64{len: n}
	for i in 0 .. n {
		indices[i] = i
		values[i] = matrix[i][i]
	}
	for i in 0 .. n {
		mut smallest := i
		for j in i + 1 .. n {
			if values[j] < values[smallest] {
				smallest = j
			}
		}
		if smallest != i {
			values[i], values[smallest] = values[smallest], values[i]
			indices[i], indices[smallest] = indices[smallest], indices[i]
		}
	}
	if compute_vectors {
		mut sorted_vectors := [][]f64{len: n, init: []f64{len: n}}
		for column in 0 .. n {
			for row in 0 .. n {
				sorted_vectors[row][column] = vectors[row][indices[column]]
			}
		}
		for row in 0 .. n {
			matrix[row] = sorted_vectors[row].clone()
		}
	}
	return values
}
