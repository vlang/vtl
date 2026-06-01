module layers

import vtl
import vtl.la

// linear_forward_f64_cpu is the CPU path for f64 linear forward (no autograd_cuda import).
fn linear_forward_f64_cpu(input &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return la.matmul[f64](input, weights.t()!)!.add[f64](bias)!
}

// linear_forward_f64 may use CUDA when enabled at runtime (`VTL_USE_CUDA=1`, `-d cuda`).
// session is an opaque pointer to autograd_cuda.DeviceSession when CUDA is enabled.
pub fn linear_forward_f64(input &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64],
	input_gpu voidptr, session voidptr) !&vtl.Tensor[f64] {
	if session != unsafe { nil } {
		if out := linear_forward_f64_session(input, weights, bias, input_gpu, session) {
			return out
		}
	}
	if linear_forward_f64_use_cuda() {
		return linear_forward_cuda_f64(input, weights, bias) or {
			return linear_forward_f64_cpu(input, weights, bias)
		}
	}
	return linear_forward_f64_cpu(input, weights, bias)
}
