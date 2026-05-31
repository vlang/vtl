module internal

import vtl

// conv2d_forward_f64 uses cuDNN when eligible (`VTL_USE_CUDA=1`, `-d cuda`), else CPU loops.
pub fn conv2d_forward_f64(input &vtl.Tensor[f64],
	weight &vtl.Tensor[f64],
	bias &vtl.Tensor[f64],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[f64] {
	if conv2d_cuda_eligible(kernel_size, config) {
		if cuda_out := conv2d_forward_cuda_f64(input, weight, bias, kernel_size, config) {
			return cuda_out
		}
	}
	return conv2d_forward_cpu_f64(input, weight, bias, kernel_size, config)
}
