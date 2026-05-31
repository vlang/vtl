module internal

import vtl

// conv2d_cuda_eligible is false without `-d cuda`.
pub fn conv2d_cuda_eligible(kernel_size []int, config Conv2DConfig) bool {
	return false
}

// conv2d_forward_cuda_f64 stub when not built with `-d cuda`.
pub fn conv2d_forward_cuda_f64(input &vtl.Tensor[f64],
	weight &vtl.Tensor[f64],
	bias &vtl.Tensor[f64],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[f64] {
	return error(@FN + ': compile with -d cuda for CUDA conv2d')
}
