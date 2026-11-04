module internal

import vtl

// layer_norm_cudnn_eligible is always false without `-d cuda`, so the CPU path runs.
fn layer_norm_cudnn_eligible() bool {
	return false
}

// layer_norm_backward_cuda_f32 exists only so the generic dispatch compiles without `-d cuda`.
fn layer_norm_backward_cuda_f32(gradient &vtl.Tensor[f32], input &vtl.Tensor[f32], gamma &vtl.Tensor[f32], eps f64, normalized_size int) ![]&vtl.Tensor[f32] {
	return error('layer_norm cuDNN backward: build with -d cuda')
}
