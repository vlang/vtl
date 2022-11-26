module la

import vsl.la
import vtl

pub fn dot[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) ?f64 {
	if !a.is_vector() || !b.is_vector() {
		return error('Tensors must be one dimensional')
	} else if a.size != b.size {
		return error('Tensors must have the same shape')
	}
	return la.vector_dot(arr_to_f64arr(a.to_array()), arr_to_f64arr(b.to_array()))
}

pub fn dger[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) ?&vtl.Tensor[T] {
	if !a.is_vector() || !b.is_vector() {
		return error('Tensors must be one dimensional')
	}
	m := la.vector_vector_tr_mul(1.0, arr_to_f64arr(a.to_array()), arr_to_f64arr(a.to_array()))
	return vtl.from_2d[f64](m.get_deep2())
}

pub fn det[T](t &vtl.Tensor[T]) ?f64 {
	t.assert_square_matrix()?
	m := t.shape[0]
	n := t.shape[1]
	mat := la.matrix_raw(m, n, arr_to_f64arr(t.to_array()))
	return mat.det()
}

pub fn inv[T](t &vtl.Tensor[T]) ?&vtl.Tensor[T] {
	t.assert_square_matrix()?
	mut colmajort := t.copy(.colmajor)
	mut ret_m := la.new_matrix[f64](colmajort.shape[0], colmajort.shape[1])
	mut colmajorm := la.matrix_raw(colmajort.shape[0], colmajort.shape[1], arr_to_f64arr(colmajort.to_array()))
	la.matrix_inv(mut ret_m, mut colmajorm, true)
	return vtl.from_2d[f64](ret_m.get_deep2())
}

pub fn matmul[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) ?&vtl.Tensor[T] {
	ma, mb := vtl.broadcast2[T](a.copy(.row_major), b.copy(.row_major))?
	mut dm := la.new_matrix[f64](a.shape[0], b.shape[1])
	mam := la.matrix_raw(a.shape[0], a.shape[1], arr_to_f64arr(ma.to_array()))
	mbm := la.matrix_raw(b.shape[0], b.shape[1], arr_to_f64arr(mb.to_array()))
	la.matrix_matrix_mul(mut dm, 1.0, mam, mbm)
	return vtl.from_2d[f64](dm.get_deep2())
}

fn arr_to_f64arr[T](a []T) []f64 {
	mut ret := []f64{cap: a.len}
	for val in a {
		ret << f64(val)
	}
	return ret
}
