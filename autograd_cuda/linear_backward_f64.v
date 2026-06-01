module autograd_cuda

import vtl
import vtl.la

// linear_backward_f64_cpu implements Linear gate gradients on CPU.
pub fn linear_backward_f64_cpu(grad &vtl.Tensor[f64], input &vtl.Tensor[f64],
	weight &vtl.Tensor[f64], bias_needs_grad bool) ![]&vtl.Tensor[f64] {
	mut result := [grad, grad, grad]
	result[0] = la.matmul[f64](grad, weight)!
	result[1] = la.matmul[f64](grad.t()!, input)!
	if bias_needs_grad {
		batch_size := grad.shape[0]
		ones := vtl.ones[f64]([1, batch_size])
		result[2] = la.matmul[f64](ones, grad)!
	}
	return result
}
