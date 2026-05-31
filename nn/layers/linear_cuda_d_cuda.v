module layers

import vtl
import vsl.cuda
import vsl.cuda.compute

// linear_forward_cuda_f64 computes y = x·Wᵀ + b using cuBLAS GEMM.
// Returns a CPU-resident tensor so autograd gates (CPU matmul) stay valid.
// Opt-in via VTL_USE_CUDA=1 and build flag `-d cuda`.
pub fn linear_forward_cuda_f64(x &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	if !cuda_linear_enabled() {
		return error('linear_forward_cuda_f64: set VTL_USE_CUDA=1 to enable')
	}
	if !x.is_matrix() || !weights.is_matrix() {
		return error('linear_forward_cuda_f64: input and weights must be matrices')
	}
	if !(bias.is_vector() || (bias.is_matrix() && bias.shape[0] == 1)) {
		return error('linear_forward_cuda_f64: bias must be vector or [1, N]')
	}

	dev := cuda.get_default_device()!

	m := x.shape[0]
	k := x.shape[1]
	n := weights.shape[0]

	x_arr := x.to_array()
	w_arr := weights.to_array()

	x_col := cuda.row_to_col_major(x_arr, m, k)
	w_col := cuda.row_to_col_major(w_arr, n, k)

	result_col := compute.gemm_cuda(dev, x_col, w_col, m, n, k)!
	mut out_row := cuda.col_to_row_major(result_col, m, n)

	b_arr := bias.to_array()
	for i in 0 .. out_row.len {
		col_idx := i % n
		out_row[i] += b_arr[col_idx]
	}

	return vtl.from_array(out_row, [m, n])!
}

// relu_forward_cuda applies ReLU on GPU and returns a CPU tensor (f64 only).
pub fn relu_forward_cuda(x &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	if !cuda_linear_enabled() {
		return error('relu_forward_cuda: set VTL_USE_CUDA=1')
	}
	dev := cuda.get_default_device()!
	input_data := x.to_array()
	result := compute.relu_cuda(dev, input_data)!
	return vtl.from_array(result, x.shape.clone())!
}

// sigmoid_forward_cuda applies Sigmoid on GPU and returns a CPU tensor (f64 only).
pub fn sigmoid_forward_cuda(x &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	if !cuda_linear_enabled() {
		return error('sigmoid_forward_cuda: set VTL_USE_CUDA=1')
	}
	dev := cuda.get_default_device()!
	input_data := x.to_array()
	result := compute.sigmoid_cuda(dev, input_data)!
	return vtl.from_array(result, x.shape.clone())!
}
