module layers

import vtl
import vtl.la

// linear_forward_f64 may use CUDA when enabled at runtime (`VTL_USE_CUDA=1`, `-d cuda`).
pub fn linear_forward_f64(input &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	if linear_forward_f64_use_cuda() {
		return linear_forward_cuda_f64(input, weights, bias) or {
			return la.matmul[f64](input, weights.t()!)!.add[f64](bias)!
		}
	}
	return la.matmul[f64](input, weights.t()!)!.add[f64](bias)!
}
