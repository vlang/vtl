module la

import vtl
import vsl.vcl
import vsl.vcl.compute

// matmul_vcl performs matrix multiplication on GPU via OpenCL.
// Both tensors must be 2D. Uses f64 precision via vsl.vcl.compute.gemm_vcl.
// Returns a Tensor[T] on the CPU.
pub fn matmul_vcl[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[T] {
	a.assert_matrix()!
	b.assert_matrix()!
	if a.shape[1] != b.shape[0] {
		return error('matmul_vcl: incompatible shapes ${a.shape} x ${b.shape}')
	}

	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[1]

	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}

	// Flatten to row-major f64 then convert to column-major for OpenCL GEMM
	a_rm := a.copy(.row_major)
	b_rm := b.copy(.row_major)
	a_f64 := a_rm.as_f64().to_array()
	b_f64 := b_rm.as_f64().to_array()

	a_col := row_to_col_major(a_f64, m, k)
	b_col := row_to_col_major(b_f64, k, n)

	c_col := compute.gemm_vcl(mut dev, a_col, b_col, m, n, k)!
	c_row := col_to_row_major(c_col, m, n)

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

	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}

	a_rm := a.copy(.row_major)
	b_rm := b.copy(.row_major)
	a_f32 := a_rm.as_f32().to_array()
	b_f32 := b_rm.as_f32().to_array()

	a_col := row_to_col_major_f32(a_f32, m, k)
	b_col := row_to_col_major_f32(b_f32, k, n)

	c_col := compute.gemm_vcl_f32(mut dev, a_col, b_col, m, n, k)!
	c_row := col_to_row_major_f32(c_col, m, n)

	mut c := vtl.from_1d[f32](c_row, vtl.TensorData{})!
	return c.reshape([m, n])!
}

// --- layout helpers ---

fn row_to_col_major(data []f64, rows int, cols int) []f64 {
	mut out := []f64{len: data.len}
	for r in 0 .. rows {
		for c in 0 .. cols {
			out[r + c * rows] = data[r * cols + c]
		}
	}
	return out
}

fn col_to_row_major(data []f64, rows int, cols int) []f64 {
	mut out := []f64{len: data.len}
	for r in 0 .. rows {
		for c in 0 .. cols {
			out[r * cols + c] = data[r + c * rows]
		}
	}
	return out
}

fn row_to_col_major_f32(data []f32, rows int, cols int) []f32 {
	mut out := []f32{len: data.len}
	for r in 0 .. rows {
		for c in 0 .. cols {
			out[r + c * rows] = data[r * cols + c]
		}
	}
	return out
}

fn col_to_row_major_f32(data []f32, rows int, cols int) []f32 {
	mut out := []f32{len: data.len}
	for r in 0 .. rows {
		for c in 0 .. cols {
			out[r * cols + c] = data[r + c * rows]
		}
	}
	return out
}
