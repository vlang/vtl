module la

import vsl.la as vsl_la
import vtl

pub fn dot[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	if !a.is_vector() || !b.is_vector() {
		return error('Tensors must be one dimensional')
	} else if a.size != b.size {
		return error('Tensors must have the same shape')
	}
	res := vsl_la.vector_dot(tensor_to_f64_array[T](a), tensor_to_f64_array[T](b))
	return vtl.from_1d([res])
}

pub fn det[T](t &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	t.assert_square_matrix()!
	m := t.shape[0]
	n := t.shape[1]
	mat := vsl_la.matrix_raw(m, n, tensor_to_f64_array[T](t))
	return vtl.from_1d([vsl_la.matrix_det(mat)])
}

pub fn inv[T](t &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	t.assert_square_matrix()!
	mut colmajort := t.copy(.col_major)
	mut ret_m := vsl_la.new_matrix[f64](colmajort.shape[0], colmajort.shape[1])
	mut colmajorm := vsl_la.matrix_raw(colmajort.shape[0], colmajort.shape[1], tensor_to_f64_array[T](colmajort))
	vsl_la.matrix_inv(mut ret_m, mut colmajorm, true)
	return vtl.from_2d[f64](ret_m.get_deep2())
}

pub fn matmul[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	a.assert_matrix()!
	b.assert_matrix()!
	if a.shape[1] != b.shape[0] {
		return error('Invalid shapes for matrix multiplication ${a.shape} and ${b.shape}')
	}
	ma, mb := vtl.broadcast2[T](a.copy(.row_major), b.copy(.row_major))!
	mut dm := vsl_la.new_matrix[f64](a.shape[0], b.shape[1])
	mam := vsl_la.matrix_raw(a.shape[0], a.shape[1], tensor_to_f64_array[T](ma))
	mbm := vsl_la.matrix_raw(b.shape[0], b.shape[1], tensor_to_f64_array[T](mb))
	vsl_la.matrix_matrix_mul(mut dm, 1.0, mam, mbm)
	return vtl.from_2d[f64](dm.get_deep2())
}

fn tensor_to_f64_array[T](t &vtl.Tensor[T]) []f64 {
	return t.as_f64().to_array()
}
