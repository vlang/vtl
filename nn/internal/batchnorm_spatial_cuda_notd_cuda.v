module internal

import vtl

// batchnorm_cudnn_eligible is always false without `-d cuda`, so the CPU path runs.
fn batchnorm_cudnn_eligible(eps f64, training bool) bool {
	return false
}

// The cuDNN backward functions exist only so the generic dispatch compiles without `-d cuda`.
fn batchnorm2d_backward_cuda_f64(gradient &vtl.Tensor[f64], input &vtl.Tensor[f64], gamma &vtl.Tensor[f64], mean &vtl.Tensor[f64], var_ &vtl.Tensor[f64], eps f64) ![]&vtl.Tensor[f64] {
	return error('batchnorm2d cuDNN backward: build with -d cuda')
}

fn batchnorm1d_backward_cuda_f64(gradient &vtl.Tensor[f64], input &vtl.Tensor[f64], gamma &vtl.Tensor[f64], mean &vtl.Tensor[f64], var_ &vtl.Tensor[f64], eps f64) ![]&vtl.Tensor[f64] {
	return error('batchnorm1d cuDNN backward: build with -d cuda')
}

fn batchnorm3d_backward_cuda_f64(gradient &vtl.Tensor[f64], input &vtl.Tensor[f64], gamma &vtl.Tensor[f64], mean &vtl.Tensor[f64], var_ &vtl.Tensor[f64], eps f64) ![]&vtl.Tensor[f64] {
	return error('batchnorm3d cuDNN backward: build with -d cuda')
}
