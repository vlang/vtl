module internal

import vtl

pub fn conv2d_backward_f64(grad_out &vtl.Tensor[f64], input &vtl.Tensor[f64], weight &vtl.Tensor[f64],
	bias &vtl.Tensor[f64], kernel_size []int, config Conv2DConfig) ![]&vtl.Tensor[f64] {
	return conv2d_backward_cpu_f64(grad_out, input, weight, bias, kernel_size, config)
}
