module autograd_cuda

import vtl
import vtl.la

// DeviceSession holds reusable staging buffers for CUDA forward ops on one Context.
// Activations remain CPU-backed for autograd; this only reduces alloc/copy overhead.

// DeviceSession defines a public data structure for this module.

// DeviceSession defines a public data structure for this module.
@[heap]
pub struct DeviceSession {
pub mut:
	enabled bool
	// Phase 2: opaque GPU activation chain (`DeviceGpuChain` in CUDA builds).
	gpu_chain voidptr = unsafe { nil }
	// Staging buffers for cuBLAS GEMM (column-major staging, row-major output)
	gemm_x_col   []f64
	gemm_w_col   []f64
	gemm_out_row []f64
	// Phase 4 (#106): opaque DeviceOptimizerState in CUDA builds.
	optimizer_state voidptr = unsafe { nil }
}

// new_device_session creates an empty session (CUDA init is build-specific).
pub fn new_device_session() &DeviceSession {
	return &DeviceSession{}
}

// new_device_session_ptr returns a session as voidptr for Context[f64] (avoids autograd↔autograd_cuda cycle).
pub fn new_device_session_ptr() voidptr {
	mut s := new_device_session()
	s.init_device()
	return unsafe { s }
}

// linear_forward_f64_cpu is the CPU fallback used by all builds.
// Uses an explicit Wᵀ buffer (weights.t() is a non-contiguous view; matmul copy is unsafe).
pub fn linear_forward_f64_cpu(x &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	n := weights.shape[0]
	k := weights.shape[1]
	wt := transpose_weights_row(weights.to_array(), n, k)
	wt_t := vtl.from_array(wt, [k, n])!
	return la.matmul[f64](x, wt_t)!.add[f64](bias)!
}

// resize_f64 grows or reuses a buffer to fit n elements.
fn resize_f64(mut buf []f64, n int) []f64 {
	if buf.len < n {
		return []f64{len: n}
	}
	return buf[..n]
}

// stage_col_major copies row-major matrix into buf as column-major (reused buffer).
fn stage_col_major(mut buf []f64, data []f64, rows int, cols int) []f64 {
	n := rows * cols
	buf = resize_f64(mut buf, n)
	mut k := 0
	for c in 0 .. cols {
		for r in 0 .. rows {
			buf[k] = data[r * cols + c]
			k++
		}
	}
	return buf
}

// transpose_weights_row builds Wᵀ as [k × n] row-major from W stored [n × k] row-major.
fn transpose_weights_row(w []f64, n int, k int) []f64 {
	mut wt := []f64{len: k * n}
	for r in 0 .. k {
		for c in 0 .. n {
			wt[r * n + c] = w[c * k + r]
		}
	}
	return wt
}

// unstage_row_major writes column-major buf into row-major slice in out.
fn unstage_row_major(col []f64, mut out []f64, rows int, cols int) {
	mut k := 0
	for c in 0 .. cols {
		for r in 0 .. rows {
			out[r * cols + c] = col[k]
			k++
		}
	}
}
