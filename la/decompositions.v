module la

import vsl.la as vsl_la
import vtl

// solve solves the linear system A * X = B for X given A and B.
// A must be square (m x m) and B has shape (m,) or (m, nrhs).
pub fn solve[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	a.assert_square_matrix()!
	b.assert_matrix()!
	if a.shape[0] != b.shape[0] {
		return error('solve: A rows (${a.shape[0]}) must match B rows (${b.shape[0]})')
	}
	n := a.shape[0]
	nrhs := if b.rank() == 1 { 1 } else { b.shape[1] }

	// Convert A to column-major flat array for VSL
	mut a_col := []f64{len: n * n}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			// vtl uses row-major storage
			a_col[i + j * n] = a.get([i, j])
		}
	}

	// Convert B to appropriate shape
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

	// Solve using VSL den_solve (overwrites a and b)
	mut x := []f64{len: n * nrhs}
	vsl_la.den_solve(mut x, b_mat, b_mat.get_deep2()[0], false)

	// Build result tensor
	if nrhs == 1 {
		return vtl.from_1d(x)
	} else {
		mut data := [][]f64{len: n}
		for i in 0 .. n {
			data[i] = x[i * nrhs..(i + 1) * nrhs]
		}
		return vtl.from_2d[f64](data)
	}
}

// lstsq solves the linear least-squares problem min ||Ax - B||_2.
// Returns (x, residuals, rank, singular_values).
pub fn lstsq[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !(&vtl.Tensor[f64], &vtl.Tensor[f64], int, &vtl.Tensor[f64]) {
	if a.rank() != 2 {
		return error('lstsq: A must be a 2D matrix')
	}
	if b.rank() > 2 {
		return error('lstsq: B must be 1D or 2D')
	}
	m := a.shape[0]
	n := a.shape[1]
	nrhs := if b.rank() == 1 { 1 } else { b.shape[1] }

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
	x_t := vtl.from_2d[f64](x)!
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
	m := a.shape[0]
	n := a.shape[1]

	mut a_mat := vsl_la.Matrix.new[f64](m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a_mat.set(i, j, a.get([i, j]))
		}
	}

	safe_tol := if tol > 0 { tol } else { 1e-8 }
	return vsl_la.rank(a_mat, safe_tol)
}

