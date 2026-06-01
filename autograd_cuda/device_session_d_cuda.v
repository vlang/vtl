module autograd_cuda

import os
import vtl
import vsl.cuda
import vsl.cuda.compute

// init_device enables staging when VTL_USE_CUDA=1.
pub fn (mut s DeviceSession) init_device() {
	if os.getenv('VTL_USE_CUDA') == '1' {
		s.enabled = true
	}
}

// linear_forward_f64 runs cuBLAS GEMM with reused session buffers; returns CPU tensor.
// When `VTL_GPU_ACTIVATIONS=1`, reuses `input_gpu` (`&CudaTensor[f64]`) to skip input H2D.
pub fn (mut s DeviceSession) linear_forward_f64(x &vtl.Tensor[f64], weights &vtl.Tensor[f64],
	bias &vtl.Tensor[f64], input_gpu voidptr) !&vtl.Tensor[f64] {
	if !s.enabled {
		return error('device session: CUDA not enabled (set VTL_USE_CUDA=1)')
	}
	if !x.is_matrix() || !weights.is_matrix() {
		return error('device session linear: input and weights must be matrices')
	}
	if !(bias.is_vector() || (bias.is_matrix() && bias.shape[0] == 1)) {
		return error('device session linear: bias must be vector or [1, N]')
	}

	if gpu_activations_enabled() {
		return s.linear_forward_gpu_chain(x, weights, bias, input_gpu)
	}

	dev := cuda.get_default_device()!

	m := x.shape[0]
	k := x.shape[1]
	n := weights.shape[0]

	x_arr := x.to_array()
	w_arr := weights.to_array()
	wt_row := transpose_weights_row(w_arr, n, k)

	mut out_row := compute.gemm_cuda(dev, x_arr, wt_row, m, n, k)!

	s.gemm_out_row = resize_f64(mut s.gemm_out_row, out_row.len)
	for i in 0 .. out_row.len {
		s.gemm_out_row[i] = out_row[i]
	}

	b_arr := bias.to_array()
	for i in 0 .. s.gemm_out_row.len {
		col_idx := i % n
		s.gemm_out_row[i] += b_arr[col_idx]
	}

	return vtl.from_array(s.gemm_out_row, [m, n])!
}
