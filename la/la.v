module la

import vsl.la
import vtl

pub fn dot(a vtl.Tensor, b vtl.Tensor) f64 {
	if !a.is_vector() || !b.is_vector() {
		panic('Tensors must be one dimensional')
	} else if a.size != b.size {
		panic('Tensors must have the same shape')
	}
	return la.vector_dot(vtl.tensor_to_varray<f64>(a), vtl.tensor_to_varray<f64>(b))
}

pub fn dger(a vtl.Tensor, b vtl.Tensor) vtl.Tensor {
	if !a.is_vector() || !b.is_vector() {
		panic('Tensors must be one dimensional')
	}
	m := la.vector_vector_tr_mul(1.0, vtl.tensor_to_varray<f64>(a), vtl.tensor_to_varray<f64>(a))
	return vtl.from_2d<f64>(m.get_deep2())
}

pub fn det(t vtl.Tensor) f64 {
	assert_square_matrix(t)
	m := t.shape[0]
	n := t.shape[1]
	mat := la.matrix_raw(m, n, vtl.tensor_to_varray<f64>(t))
	return mat.det()
}

pub fn inv(t vtl.Tensor) vtl.Tensor {
	assert_square_matrix(t)
	mut colmajort := t.copy(.colmajor)
	mut ret_m := la.new_matrix(colmajort.shape[0], colmajort.shape[1])
	colmajorm := la.matrix_raw(colmajort.shape[0], colmajort.shape[1], vtl.tensor_to_varray<f64>(colmajort))
	la.matrix_inv(mut ret_m, colmajorm, true)
	return vtl.from_2d<f64>(ret_m.get_deep2())
}

pub fn matmul(a vtl.Tensor, b vtl.Tensor) vtl.Tensor {
	ma := match a.is_contiguous() {
		true { a }
		else { a.copy(.rowmajor) }
	}
	mb := match b.is_contiguous() {
		true { b }
		else { b.copy(.rowmajor) }
	}
	mut dm := la.new_matrix(a.shape[0], b.shape[1])
	mam := la.matrix_raw(a.shape[0], a.shape[1], vtl.tensor_to_varray<f64>(ma))
	mbm := la.matrix_raw(b.shape[0], b.shape[1], vtl.tensor_to_varray<f64>(mb))
	la.matrix_matrix_mul(mut dm, 1.0, mam, mbm)
	return vtl.from_2d<f64>(dm.get_deep2())
}

pub fn tensordot(a vtl.Tensor, b vtl.Tensor, a_axes []int, b_axes []int) vtl.Tensor {
	outshape, a_newshape, b_newshape, a_newaxes, b_newaxes := tensordot_output_data(a,
		b, a_axes, b_axes)
	at := a.transpose(a_newaxes).reshape(a_newshape)
	bt := b.transpose(b_newaxes).reshape(b_newshape)
	res := matmul(at, bt)
	return res.reshape(outshape)
}
