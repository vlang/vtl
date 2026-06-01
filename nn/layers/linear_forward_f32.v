module layers

import vtl
import vtl.la

// linear_forward_f32 uses Vulkan GEMM when `VTL_USE_VULKAN=1` and `-d vulkan`, else CPU.
pub fn linear_forward_f32(input &vtl.Tensor[f32], weights &vtl.Tensor[f32], bias &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	if linear_forward_f32_use_vulkan() {
		return linear_forward_vulkan_f32(input, weights, bias) or {
			return linear_forward_f32_cpu(input, weights, bias)
		}
	}
	return linear_forward_f32_cpu(input, weights, bias)
}

fn linear_forward_f32_cpu(input &vtl.Tensor[f32], weights &vtl.Tensor[f32], bias &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return la.matmul[f32](input, weights.t()!)!.add[f32](bias)!
}
