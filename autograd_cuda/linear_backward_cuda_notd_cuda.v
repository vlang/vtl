module autograd_cuda

import vtl

pub fn linear_backward_f64(grad &vtl.Tensor[f64], input &vtl.Tensor[f64], weight &vtl.Tensor[f64],
	bias_needs_grad bool, mut session DeviceSession) ![]&vtl.Tensor[f64] {
	_ = session
	return linear_backward_f64_cpu(grad, input, weight, bias_needs_grad)
}
