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

pub fn ddot(a num.NdArray, b num.NdArray) f64 {
	if a.ndims != 1 || b.ndims != 1 {
		panic('Tensors must be one dimensional')
	} else if a.size != b.size {
		panic('Tensors must have the same shape')
	}
	return blas.ddot(a.size, a.f64_array(), a.strides[0], b.f64_array(), b.strides[0])
}

pub fn dger(a num.NdArray, mut b num.NdArray) num.NdArray {
	if a.ndims != 1 || b.ndims != 1 {
		panic('Tensors must be one dimensional')
	}
	out := num.empty([a.size, b.size])
        mut mut_b := b.f64_array()
	blas.dger(a.size, b.size, 1.0, a.f64_array(), a.strides[0],
		mut mut_b, b.strides[0], out.f64_array(), out.shape[1])
        b.assign(num.from_f64(mut_b, b.shape))
	return out
}

pub fn dnrm2(a num.NdArray) f64 {
	if a.ndims != 1 {
		panic('Tensor must be one dimensional')
	}
	return blas.dnrm2(a.size, a.f64_array(), a.strides[0])
}

pub fn dlange(a num.NdArray, norm byte) f64 {
	if a.ndims != 2 {
		panic('Tensor must be two-dimensional')
	}
	m := fortran_view_or_copy(a)
	work := []f64{len: m.shape[0] * int(sizeof(f64))}
	return blas.dlange(norm, m.shape[0], m.shape[1], m.f64_array(), m.shape[0], work)
}

pub fn dpotrf(a num.NdArray, up bool) num.NdArray {
	if a.ndims != 2 {
		panic('Tensor must be two-dimensional')
	}
	mut ret := a.copy('F')
        mut mut_ret := ret.f64_array()
	blas.dpotrf(up, ret.shape[0], mut mut_ret, ret.shape[0])
        ret.assign(num.from_f64(mut_ret, ret.shape))
	if up {
		num.triu_inpl(mut ret)
	} else {
		num.tril_inpl(mut ret)
	}
	return ret
}

pub fn det(a num.NdArray) f64 {
	ret := a.copy('F')
	m := a.shape[0]
	n := a.shape[1]
	ipiv := []int{len: (int(sizeof(int)) * n)}
        mut mut_a := a.f64_array()
	blas.dgetrf(m, n, mut mut_a, m, ipiv)
        a.assign(num.from_f64(mut_a, a.shape))
	ldet := num.prod(ret.diagonal())
	mut detp := 1
	for i := 0; i < n; i++ {
		if (i + 1) != ipiv[i] {
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
	ipiv := []int{len: (n * int(sizeof(int)))}
	mut mut_ret := ret.f64_array()
	blas.dgetrf(n, n, mut mut_ret, n, ipiv)
	blas.dgetri(n, mut mut_ret, n, ipiv)
        ret.assign(num.from_f64(mut_ret, ret.shape))
	return ret
}

pub fn matmul(a num.NdArray, b num.NdArray) num.NdArray {
	mut dest := []f64{len: a.shape[0] * b.shape[1]}
	ma := match a.flags.contiguous {
		true { a }
		else { a.copy('C') }
	}
	mb := match b.flags.contiguous {
		true { b }
		else { b.copy('C') }
	}
	blas.dgemm(false, false, ma.shape[0],
		mb.shape[1], ma.shape[1], 1.0, ma.f64_array(), ma.shape[1], mb.f64_array(), mb.shape[1], 1.0,
		mut dest, mb.shape[1])
	return num.from_f64(dest, [a.shape[0], b.shape[1]])
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

pub fn solve(a num.NdArray, b num.NdArray) num.NdArray {
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
	mut ret := a.copy('F')
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
	num.triu_inpl_offset(mut ret, -1)
	return ret
}
