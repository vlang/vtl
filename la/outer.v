module la

import vsl.la as vsl_la
import vtl

// outer computes the outer product of two vectors.
// result[i,j] = u[i] * v[j]
// result shape: [u.len, v.len]
pub fn outer[T](u &vtl.Tensor[T], v &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	if u.rank() != 1 || v.rank() != 1 {
		return error('outer: both inputs must be 1D vectors')
	}
	ul := u.shape[0]
	vl := v.shape[0]
	_ = ul
	_ = vl
	ua := tensor_to_f64_array[T](u)
	va := tensor_to_f64_array[T](v)
	mat := vsl_la.outer(ua, va)
	return vtl.from_2d[f64](mat.get_deep2())
}

// cross computes the 3D cross product of two vectors: u × v
// Both vectors must have length 3.
pub fn cross[T](u &vtl.Tensor[T], v &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	if u.rank() != 1 || v.rank() != 1 {
		return error('cross: both inputs must be 1D vectors')
	}
	if u.shape[0] != 3 || v.shape[0] != 3 {
		return error('cross: both vectors must have length 3')
	}
	ua := tensor_to_f64_array[T](u)
	va := tensor_to_f64_array[T](v)
	res := vsl_la.cross(ua, va)
	return vtl.from_1d(res)
}

