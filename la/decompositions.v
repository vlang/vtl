module la

import vsl.la as vsl_la
import vsl.lapack as vsl_lapack
import vtl

// solve solves A * X = B for stacks of square A matrices. Leading dimensions
// broadcast like NumPy; B may be a vector or a matrix of right-hand sides.
pub fn solve[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	if a.rank() < 2 || b.rank() < 1 {
		return error('solve: A must be a matrix stack and B must have at least one dimension')
	}
	n := a.shape[a.rank() - 2]
	if a.shape[a.rank() - 1] != n {
		return error('solve: A matrices must be square')
	}
	b_is_vector := b.rank() == 1 || (a.rank() > 2 && b.rank() == a.rank() - 1)
	b_rows := if b_is_vector { b.shape[b.rank() - 1] } else { b.shape[b.rank() - 2] }
	nrhs := if b_is_vector { 1 } else { b.shape[b.rank() - 1] }
	if b_rows != n {
		return error('solve: A dimension ${n} does not match B rows ${b_rows}')
	}
	a_batch_shape := a.shape[..a.rank() - 2]
	b_batch_shape := if b_is_vector && b.rank() > 1 {
		b.shape[..b.rank() - 1]
	} else if b_is_vector {
		[]int{}
	} else {
		b.shape[..b.rank() - 2]
	}
	batch_shape := matmul_broadcast_shape(a_batch_shape, b_batch_shape) or {
		return error('solve: batch shapes ${a_batch_shape} and ${b_batch_shape} cannot broadcast')
	}
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	mut output_shape := batch_shape.clone()
	output_shape << n
	if !b_is_vector {
		output_shape << nrhs
	}
	mut output := vtl.empty[f64](output_shape, memory: .row_major)
	if n == 0 || nrhs == 0 {
		return output
	}
	mut a_index := []int{len: a.rank()}
	mut b_index := []int{len: b.rank()}
	for batch in 0 .. batch_count {
		batch_coordinates := decode_batch_coordinates(batch, batch_shape)
		fill_broadcast_batch_index(a_batch_shape, batch_shape, batch_coordinates, mut a_index)
		if !b_is_vector {
			fill_broadcast_batch_index(b_batch_shape, batch_shape, batch_coordinates, mut b_index)
		} else if b.rank() > 1 {
			fill_broadcast_batch_index(b_batch_shape, batch_shape, batch_coordinates, mut b_index)
		}
		mut matrix := []f64{len: n * n}
		mut rhs := []f64{len: n * nrhs}
		for row in 0 .. n {
			a_index[a.rank() - 2] = row
			for column in 0 .. n {
				a_index[a.rank() - 1] = column
				matrix[row * n + column] = f64(a.get[T](a_index))
			}
			if b_is_vector {
				b_index[b.rank() - 1] = row
				rhs[row * nrhs] = f64(b.get[T](b_index))
			} else {
				b_index[b.rank() - 2] = row
				for column in 0 .. nrhs {
					b_index[b.rank() - 1] = column
					rhs[row * nrhs + column] = f64(b.get[T](b_index))
				}
			}
		}
		mut pivots := []int{len: n}
		info := vsl_lapack.dgesv(n, nrhs, mut matrix, n, mut pivots, mut rhs, nrhs)
		if info < 0 {
			return error('solve: LAPACK rejected argument ${-info}')
		}
		if info > 0 {
			return error('solve: matrix is singular')
		}
		start := batch * n * nrhs
		for row in 0 .. n {
			for column in 0 .. nrhs {
				output.set_nth(start + row * nrhs + column, rhs[row * nrhs + column])
			}
		}
	}
	return output
}

fn decode_batch_coordinates(batch int, shape []int) []int {
	mut coordinates := []int{len: shape.len}
	mut remainder := batch
	for i := shape.len - 1; i >= 0; i-- {
		coordinates[i] = remainder % shape[i]
		remainder /= shape[i]
	}
	return coordinates
}

fn fill_broadcast_batch_index(source_shape []int, output_shape []int, output_coordinates []int, mut index []int) {
	offset := output_shape.len - source_shape.len
	for i, dimension in source_shape {
		index[i] = if dimension == 1 { 0 } else { output_coordinates[offset + i] }
	}
}

// lstsq solves the linear least-squares problem min ||Ax - B||_2.
// Returns (x, residuals, rank, singular_values).
pub fn lstsq[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !(&vtl.Tensor[f64], &vtl.Tensor[f64], int, &vtl.Tensor[f64]) {
	if a.rank() != 2 {
		return error('lstsq: A must be a 2D matrix')
	}
	if b.rank() < 1 || b.rank() > 2 {
		return error('lstsq: B must be 1D or 2D')
	}
	m := a.shape[0]
	n := a.shape[1]
	if b.shape[0] != m {
		return error('lstsq: A rows (${m}) must match B rows (${b.shape[0]})')
	}
	b_is_vector := b.rank() == 1
	nrhs := if b_is_vector { 1 } else { b.shape[1] }

	// Build VSL matrices
	mut a_mat := vsl_la.Matrix.new[f64](m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a_mat.set(i, j, a.get([i, j]))
		}
	}
	mut b_mat := vsl_la.Matrix.new[f64](b.shape[0], nrhs)
	for i := 0; i < b.shape[0]; i++ {
		for j := 0; j < nrhs; j++ {
			if b.rank() == 1 {
				b_mat.set(i, j, b.get([i]))
			} else {
				b_mat.set(i, j, b.get([i, j]))
			}
		}
	}

	x, residuals, rnk, s := vsl_la.lstsq(a_mat, b_mat)

	// Convert x back to vtl tensor
	x_t := if b_is_vector {
		mut vector := []f64{len: n}
		for i in 0 .. n {
			vector[i] = x[i][0]
		}
		vtl.from_1d[f64](vector)!
	} else {
		vtl.from_2d[f64](x)!
	}
	res_t := vtl.from_1d(residuals)!
	s_t := vtl.from_1d(s)!

	return x_t, res_t, rnk, s_t
}

// qr computes QR factorization of A.
// Returns (Q, R) where Q is orthonormal and R is upper triangular.
// Q shape: [m, min(m,n)], R shape: [min(m,n), n]
pub fn qr[T](a &vtl.Tensor[T]) !(&vtl.Tensor[f64], &vtl.Tensor[f64]) {
	a.assert_matrix()!
	m := a.shape[0]
	n := a.shape[1]

	mut a_mat := vsl_la.Matrix.new[f64](m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a_mat.set(i, j, a.get([i, j]))
		}
	}

	q_mat, r_mat := vsl_la.qr(a_mat)!

	q_rows := q_mat.m
	q_cols := q_mat.n
	mut q_data := [][]f64{len: q_rows, init: []f64{len: q_cols}}
	for i := 0; i < q_rows; i++ {
		for j := 0; j < q_cols; j++ {
			q_data[i][j] = q_mat.get(i, j)
		}
	}

	r_rows := r_mat.m
	r_cols := r_mat.n
	mut r_data := [][]f64{len: r_rows, init: []f64{len: r_cols}}
	for i := 0; i < r_rows; i++ {
		for j := 0; j < r_cols; j++ {
			r_data[i][j] = r_mat.get(i, j)
		}
	}

	q_t := vtl.from_2d[f64](q_data)!
	r_t := vtl.from_2d[f64](r_data)!

	return q_t, r_t
}

// lu computes LU decomposition with partial pivoting: PA = LU.
// Returns (L, U, piv) as 2D tensors.
// L shape: [m, min(m,n)], U shape: [min(m,n), n]
pub fn lu[T](a &vtl.Tensor[T]) !(&vtl.Tensor[f64], &vtl.Tensor[f64], &vtl.Tensor[int]) {
	a.assert_matrix()!
	m := a.shape[0]
	n := a.shape[1]

	mut a_mat := vsl_la.Matrix.new[f64](m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a_mat.set(i, j, a.get([i, j]))
		}
	}

	l_mat, u_mat, ipiv := vsl_la.lu(a_mat)!

	l_rows := l_mat.m
	l_cols := l_mat.n
	mut l_data := [][]f64{len: l_rows, init: []f64{len: l_cols}}
	for i := 0; i < l_rows; i++ {
		for j := 0; j < l_cols; j++ {
			l_data[i][j] = l_mat.get(i, j)
		}
	}

	u_rows := u_mat.m
	u_cols := u_mat.n
	mut u_data := [][]f64{len: u_rows, init: []f64{len: u_cols}}
	for i := 0; i < u_rows; i++ {
		for j := 0; j < u_cols; j++ {
			u_data[i][j] = u_mat.get(i, j)
		}
	}

	l_t := vtl.from_2d[f64](l_data)!
	u_t := vtl.from_2d[f64](u_data)!
	piv_t := vtl.from_1d(ipiv)!

	return l_t, u_t, piv_t
}

// cholesky computes Cholesky factorization of a symmetric positive-definite matrix.
// Returns lower-triangular L where A = L * L^T.
pub fn cholesky[T](a &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	a.assert_square_matrix()!
	n := a.shape[0]

	mut a_mat := vsl_la.Matrix.new[f64](n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			a_mat.set(i, j, a.get([i, j]))
		}
	}

	vsl_la.potrf(mut a_mat, .lower)!

	// Extract lower triangular part
	mut l_data := [][]f64{len: n, init: []f64{len: n}}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			l_data[i][j] = a_mat.get(i, j)
		}
	}

	return vtl.from_2d[f64](l_data)
}

// pinv computes the Moore-Penrose pseudoinverse of A using SVD.
pub fn pinv[T](a &vtl.Tensor[T], tol f64) !&vtl.Tensor[f64] {
	m := a.shape[0]
	n := a.shape[1]

	mut a_mat := vsl_la.Matrix.new[f64](m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a_mat.set(i, j, a.get([i, j]))
		}
	}

	// SVD
	mut s := []f64{len: int_min(m, n)}
	mut u_mat := vsl_la.Matrix.new[f64](m, m)
	mut vt_mat := vsl_la.Matrix.new[f64](n, n)
	vsl_la.matrix_svd(mut s, mut u_mat, mut vt_mat, mut a_mat, true)

	// Pseudo-inverse: V * Σ⁻¹ * U^T
	safe_tol := if tol > 0 { tol } else { 1e-8 }
	mut pinv_mat := vsl_la.Matrix.new[f64](n, m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			mut sum := 0.0
			for k := 0; k < int_min(m, n); k++ {
				if s[k] > safe_tol {
					sum += vt_mat.get(k, i) * u_mat.get(j, k) / s[k]
				}
			}
			pinv_mat.set(i, j, sum)
		}
	}

	mut data := [][]f64{len: n, init: []f64{len: m}}
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			data[i][j] = pinv_mat.get(i, j)
		}
	}
	return vtl.from_2d[f64](data)
}

// matrix_rank returns the effective numerical rank of A.
pub fn matrix_rank[T](a &vtl.Tensor[T], tol f64) !int {
	if a.rank() != 2 {
		return error('matrix_rank requires a 2D matrix')
	}
	rows := a.shape[0]
	columns := a.shape[1]
	values := svdvals[T](a)!
	mut largest := 0.0
	if values.size > 0 {
		largest = values.get_nth(0)
	}
	threshold := matrix_rank_threshold[T](tol, largest, rows, columns)
	mut rank := 0
	for value in values.to_array() {
		if value > threshold {
			rank++
		}
	}
	return rank
}

// matrix_rank_batch returns the numerical rank of each trailing matrix. A
// positive tol is an absolute threshold; non-positive tol selects NumPy's
// dtype-aware default threshold.
pub fn matrix_rank_batch[T](input &vtl.Tensor[T], tol f64) !&vtl.Tensor[int] {
	if input.rank() < 2 {
		return error('matrix_rank_batch requires input with rank at least 2')
	}
	rows := input.shape[input.rank() - 2]
	columns := input.shape[input.rank() - 1]
	values := svdvals[T](input)!
	value_count := values.shape[values.rank() - 1]
	batch_shape := input.shape[..input.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	output_shape := if batch_shape.len == 0 { [1] } else { batch_shape.clone() }
	mut result := vtl.empty[int](output_shape, memory: .row_major)
	for batch in 0 .. batch_count {
		largest := if value_count == 0 { 0.0 } else { values.get_nth(batch * value_count) }
		threshold := matrix_rank_threshold[T](tol, largest, rows, columns)
		mut rank := 0
		for i in 0 .. value_count {
			if values.get_nth(batch * value_count + i) > threshold {
				rank++
			}
		}
		result.set_nth(batch, rank)
	}
	return result
}

fn matrix_rank_threshold[T](tol f64, largest f64, rows int, columns int) f64 {
	if tol > 0 {
		return tol
	}
	mut epsilon := 2.220446049250313e-16
	$if T is f32 {
		epsilon = 1.1920928955078125e-7
	}
	return largest * f64(int_max(rows, columns)) * epsilon
}
