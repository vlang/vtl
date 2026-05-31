module layers

import vtl
import vtl.autograd

// linear_forward_f64 may use CUDA when enabled at runtime (`VTL_USE_CUDA=1`, `-d cuda`).
// When session is non-nil and initialized, GEMM staging buffers are reused (issue #91).
pub fn linear_forward_f64(input &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64],
	input_gpu voidptr, mut session autograd.DeviceSession) !&vtl.Tensor[f64] {
	if session != unsafe { nil } {
		if out := session.linear_forward_f64(input, weights, bias, input_gpu) {
			return out
		}
	}
	if linear_forward_f64_use_cuda() {
		return linear_forward_cuda_f64(input, weights, bias) or {
			return autograd.linear_forward_f64_cpu(input, weights, bias)
		}
	}
	return autograd.linear_forward_f64_cpu(input, weights, bias)
}
