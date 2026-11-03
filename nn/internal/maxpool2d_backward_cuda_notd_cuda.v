module internal

import vtl

// maxpool2d_cudnn_eligible is always false without `-d cuda`, so the CPU path runs.
fn maxpool2d_cudnn_eligible(input &vtl.Tensor[f64], max_indices &vtl.Tensor[int], grad_output &vtl.Tensor[f64], kernel []int, stride []int, padding []int) bool {
	return false
}

// maxpool2d_backward_cuda_f64 exists only so the generic dispatch compiles without `-d cuda`.
fn maxpool2d_backward_cuda_f64(shape []int, max_indices &vtl.Tensor[int], input &vtl.Tensor[f64], grad_output &vtl.Tensor[f64], kernel []int, stride []int) !&vtl.Tensor[f64] {
	return error('maxpool2d cuDNN backward: build with -d cuda')
}
