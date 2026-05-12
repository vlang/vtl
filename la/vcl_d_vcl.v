module la

import vtl
import vsl.compute as vsl_compute

// matmul_vcl performs matrix multiplication via the unified VSL compute API
// with VCL backend selected.
pub fn matmul_vcl[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[T] {
	a.assert_matrix()!
	b.assert_matrix()!
	if a.shape[1] != b.shape[0] {
		return error('matmul_vcl: incompatible shapes ${a.shape} x ${b.shape}')
	}

	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[1]

	cctx := vsl_compute.new_context(.vcl)
	a_f64 := a.copy(.row_major).as_f64().to_array()
	b_f64 := b.copy(.row_major).as_f64().to_array()
	c_row := vsl_compute.gemm(cctx, a_f64, b_f64, m, n, k)!

	c_data := c_row.map(vtl.cast[T](it))
	mut c := vtl.from_1d[T](c_data, vtl.TensorData{})!
	return c.reshape([m, n])!
}

// matmul_vcl_f32 is the f32 specialization of matmul_vcl.
pub fn matmul_vcl_f32(a &vtl.Tensor[f32], b &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	a.assert_matrix()!
	b.assert_matrix()!
	if a.shape[1] != b.shape[0] {
		return error('matmul_vcl_f32: incompatible shapes ${a.shape} x ${b.shape}')
	}

	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[1]

	cctx := vsl_compute.new_context(.vcl)
	a_f64 := a.copy(.row_major).as_f64().to_array()
	b_f64 := b.copy(.row_major).as_f64().to_array()
	c_row_f64 := vsl_compute.gemm(cctx, a_f64, b_f64, m, n, k)!
	c_row := c_row_f64.map(f32(it))

	mut c := vtl.from_1d[f32](c_row, vtl.TensorData{})!
	return c.reshape([m, n])!
}
