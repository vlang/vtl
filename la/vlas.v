module la

import vsl.la
import vtl

pub fn dot<T>(a &vtl.Tensor<T>, b &vtl.Tensor<T>) f64 {
	if !a.is_vector() || !b.is_vector() {
		panic('Tensors must be one dimensional')
	} else if a.size != b.size {
		panic('Tensors must have the same shape')
	}
	return la.vector_dot(a.to_array(), b.to_array())
}

pub fn dger<T>(a &vtl.Tensor<T>, b &vtl.Tensor<T>) &vtl.Tensor<T> {
	if !a.is_vector() || !b.is_vector() {
		panic('Tensors must be one dimensional')
	}
	m := la.vector_vector_tr_mul(1.0, a.to_array(), a.to_array())
	return vtl.from_2d<f64>(m.get_deep2())
}

pub fn det<T>(t &vtl.Tensor<T>) f64 {
	vtl.assert_square_matrix(t)
	m := t.shape[0]
	n := t.shape[1]
	mat := la.matrix_raw(m, n, t.to_array())
	return mat.det()
}

pub fn inv<T>(t &vtl.Tensor<T>) &vtl.Tensor<T> {
	vtl.assert_square_matrix(t)
	mut colmajort := t.copy(.colmajor)
	mut ret_m := la.new_matrix(colmajort.shape[0], colmajort.shape[1])
	mut colmajorm := la.matrix_raw(colmajort.shape[0], colmajort.shape[1], colmajort.to_array())
	la.matrix_inv(mut ret_m, mut colmajorm, true)
	return vtl.from_2d<f64>(ret_m.get_deep2())
}

pub fn matmul<T>(a &vtl.Tensor<T>, b &vtl.Tensor<T>) &vtl.Tensor<T> {
	ma := a.copy(.row_major)
	mb := b.copy(.row_major)
	mut dm := la.new_matrix(a.shape[0], b.shape[1])
	mam := la.matrix_raw(a.shape[0], a.shape[1], ma.to_array())
	mbm := la.matrix_raw(b.shape[0], b.shape[1], mb.to_array())
	la.matrix_matrix_mul(mut dm, 1.0, mam, mbm)
	return vtl.from_2d<f64>(dm.get_deep2())
}
