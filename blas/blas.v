module blas

import vnum.num
import vsl.blas

pub struct Workspace {
	size int
	work &f64
}

pub fn allocate_workspace(size int) Workspace {
	ptr := &f64(v_calloc(size * int(int(sizeof(f64)))))
	return Workspace{
		size: size
		work: ptr
	}
}

pub fn fortran_view_or_copy(t num.NdArray) num.NdArray {
	if t.flags.fortran {
		return t.view()
	} else {
		return t.copy('F')
	}
}

pub fn fortran_copy(t num.NdArray) num.NdArray {
	return t.copy('F')
}

pub fn assert_square_matrix(a num.NdArray) {
	if a.ndims != 2 || a.shape[0] != a.shape[1] {
		panic('Matrix is not square')
	}
}

pub fn assert_matrix(a num.NdArray) {
	if a.ndims != 2 {
		panic('Tensor is not two-dimensional')
	}
}

pub fn ddot(a, b num.NdArray) f64 {
	if a.ndims != 1 || b.ndims != 1 {
		panic('Tensors must be one dimensional')
	} else if a.size != b.size {
		panic('Tensors must have the same shape')
	}
	return C.cblas_ddot(a.size, a.buffer(), a.strides[0], b.buffer(), b.strides[0])
}

pub fn dger(a, b num.NdArray) num.NdArray {
	if a.ndims != 1 || b.ndims != 1 {
		panic('Tensors must be one dimensional')
	}
	out := num.empty([a.size, b.size])
	C.cblas_dger(blas.lapack_row_major, a.size, b.size, 1.0, a.buffer(), a.strides[0],
		b.buffer(), b.strides[0], out.buffer(), out.shape[1])
	return out
}

pub fn dnrm2(a num.NdArray) f64 {
	if a.ndims != 1 {
		panic('Tensor must be one dimensional')
	}
	return C.cblas_dnrm2(a.size, a.buffer(), a.strides[0])
}

pub fn dlange(a num.NdArray, norm byte) f64 {
	if a.ndims != 2 {
		panic('Tensor must be two-dimensional')
	}
	m := fortran_view_or_copy(a)
	work := &f64(v_calloc(m.shape[0] * int(sizeof(f64))))
	return C.LAPACKE_dlange(norm, &m.shape[0], &m.shape[1], m.buffer(), &m.shape[0], work)
}

pub fn dpotrf(a num.NdArray, up bool) num.NdArray {
	if a.ndims != 2 {
		panic('Tensor must be two-dimensional')
	}
	ret := a.copy('F')
	info := 0
	C.LAPACKE_dpotrf(blas.l_uplo(up), &ret.shape[0], ret.buffer(), &ret.shape[0], &info)
	if info > 0 {
		panic('Tensor is not positive definite')
	}
	if up {
		num.triu_inpl(ret)
	} else {
		num.tril_inpl(ret)
	}
	return ret
}

pub fn det(a num.NdArray) f64 {
	ret := a.copy('F')
	m := a.shape[0]
	n := a.shape[1]
	ipiv := &int(v_calloc(int(sizeof(int)) * n))
	info := 0
	C.LAPACKE_dgetrf(&m, &n, ret.buffer(), &m, ipiv, &info)
	if info > 0 {
		panic('Singular matrix')
	}
	ldet := num.prod(ret.diagonal())
	mut detp := 1
	for i := 0; i < n; i++ {
		if (i + 1) != unsafe {*(ipiv + i)} {
			detp = -detp
		}
	}
	return ldet * detp
}

pub fn inv(a num.NdArray) num.NdArray {
	if a.ndims != 2 || a.shape[0] != a.shape[1] {
		panic('Matrix must be square')
	}
	ret := a.copy('F')
	n := a.shape[0]
	ipiv := &int(v_calloc(n * int(sizeof(int))))
	mut info := 0
	C.LAPACKE_dgetrf(&n, &n, ret.buffer(), &n, ipiv, &info)
	if info > 0 {
		panic('Singular matrix')
	}
	lwork := n * n
	work := &f64(v_calloc(lwork * int(sizeof(f64))))
	info = C.LAPACKE_dgetri(&n, ret.buffer(), &n, ipiv, &work)
	if info != 0 {
		panic('lapack failed')
	}
	return ret
}

pub fn matmul(a, b num.NdArray) num.NdArray {
	dest := num.empty([a.shape[0], b.shape[1]])
	ma := match a.flags.contiguous {
		true { a }
		else { a.copy('C') }
	}
	mb := match b.flags.contiguous {
		true { b }
		else { b.copy('C') }
	}
	C.cblas_dgemm(blas.cblas_row_major, blas.c_trans(false), blas.c_trans(false), ma.shape[0],
		mb.shape[1], ma.shape[1], 1.0, ma.buffer(), ma.shape[1], mb.buffer(), mb.shape[1], 1.0,
		dest.buffer(), dest.shape[1])
	return dest
}

pub fn eigh(a num.NdArray) []num.NdArray {
	assert_square_matrix(a)
	ret := a.copy('F')
	n := ret.shape[0]
	w := num.empty([n])
	jobz := blas.job_vlr(true)
	uplo := blas.l_uplo(false)
	info := 0
	workspace := allocate_workspace(3 * n - 1)
	C.LAPACKE_dsyev(jobz, uplo, &n, ret.buffer(), &n, w.buffer(), workspace.work)
	if info > 0 {
		panic('Failed to converge')
	}
	return [w, ret]
}

pub fn eig(a num.NdArray) []num.NdArray {
	assert_square_matrix(a)
	ret := a.copy('F')
	n := ret.shape[0]
	wr := num.empty([n])
	wl := wr.copy('C')
	vl := num.allocate_cpu([n, n], 'F')
	vr := vl.copy('C')
	workspace := allocate_workspace(n * 4)
	jobvr := blas.job_vlr(true)
	jobvl := blas.job_vlr(true)
	info := 0
	C.LAPACKE_dgeev(jobvl, jobvr, &n, ret.buffer(), &n, wr.buffer(), wl.buffer(), vl.buffer(),
		&n, vr.buffer(), &n, workspace.work)
	if info > 0 {
		panic('QR algorithm failed')
	}
	return [wr, vl]
}

pub fn eigvalsh(a num.NdArray) num.NdArray {
	assert_square_matrix(a)
	ret := fortran_view_or_copy(a)
	n := ret.shape[0]
	jobz := blas.job_vlr(true)
	uplo := blas.l_uplo(false)
	info := 0
	w := num.empty([n])
	workspace := allocate_workspace(3 * n - 1)
	C.LAPACKE_dsyev(jobz, uplo, &n, ret.buffer(), &n, w.buffer(), workspace.work)
	if info > 0 {
		panic('Failed to converge')
	}
	return w
}

pub fn eigvals(a num.NdArray) num.NdArray {
	assert_square_matrix(a)
	ret := a.copy('F')
	n := ret.shape[0]
	wr := num.empty([n])
	wl := wr.copy('C')
	vl := num.allocate_cpu([n, n], 'F')
	vr := vl.copy('C')
	workspace := allocate_workspace(n * 3)
	jobvr := blas.job_vlr(false)
	jobvl := blas.job_vlr(false)
	info := 0
	C.LAPACKE_dgeev(jobvl, jobvr, &n, ret.buffer(), &n, wr.buffer(), wl.buffer(), vl.buffer(),
		&n, vr.buffer(), &n, workspace.work)
	if info > 0 {
		panic('QR algorithm failed')
	}
	return wr
}

pub fn solve(a, b num.NdArray) num.NdArray {
	assert_square_matrix(a)
	af := fortran_view_or_copy(a)
	bf := b.copy('F')
	n := af.shape[0]
	mut m := bf.shape[0]
	if bf.ndims > 1 {
		m = bf.shape[1]
	}
	ipiv := &int(v_calloc(n * int(sizeof(int))))
	info := 0
	C.LAPACKE_dgesv(&n, &m, af.buffer(), &n, ipiv, bf.buffer(), &m, &info)
	return bf
}

pub fn hessenberg(a num.NdArray) num.NdArray {
	assert_square_matrix(a)
	ret := a.copy('F')
	if ret.shape[0] < 2 {
		return ret
	}
	n := ret.shape[0]
	s := num.empty([n])
	ilo := 0
	ihi := 0
	job := `B`
	info := 0
	C.LAPACKE_dgebal(job, &n, ret.buffer(), &n, &ilo, &ihi, s.buffer(), &info)
	tau := num.empty([n])
	workspace := allocate_workspace(n)
	C.LAPACKE_dgehrd(n, &ilo, &ihi, ret.buffer(), &n, tau.buffer(), workspace.work)
	num.triu_inpl_offset(ret, -1)
	return ret
}
