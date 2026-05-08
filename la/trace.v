module la

import vsl.la as vsl_la
import vtl

// trace returns the sum of diagonal elements of a square matrix.
pub fn trace[T](t &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	t.assert_square_matrix()!
	m := t.shape[0]
	mat := vsl_la.Matrix.raw(m, m, tensor_to_f64_array[T](t))
	return vtl.from_1d([vsl_la.trace(mat)])
}

// norm returns the matrix norm of a tensor.
// ord: "F" (Frobenius, default), "1" (column sum), "I" (row sum / infinity).
pub fn norm[T](t &vtl.Tensor[T], ord string) !&vtl.Tensor[f64] {
	if t.rank() != 2 {
		return error('norm: tensor must be 2D (matrix), got rank ${t.rank()}')
	}
	m := t.shape[0]
	n := t.shape[1]
	mat := vsl_la.Matrix.raw(m, n, tensor_to_f64_array[T](t))
	return vtl.from_1d([vsl_la.norm(mat, ord)])
}

fn tensor_to_f64_array[T](t &vtl.Tensor[T]) []f64 {
	return t.as_f64().to_array()
}