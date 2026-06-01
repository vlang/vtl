module internal

import os
import vtl
import vtl.la
import vtl.storage

// linear_backward_vulkan_f32 computes Linear gate gradients with Vulkan GEMM (f32).
pub fn linear_backward_vulkan_f32(grad &vtl.Tensor[f32], input &vtl.Tensor[f32],
	weight &vtl.Tensor[f32], bias_needs_grad bool) ![]&vtl.Tensor[f32] {
	if os.getenv('VTL_USE_VULKAN') != '1' {
		return error('linear_backward_vulkan_f32: set VTL_USE_VULKAN=1')
	}
	mut g_vk := grad.vulkan()!
	defer { g_vk.release() }
	vk_params := storage.vulkan_params_for_device(g_vk.data.data.device)
	mut w_vk := weight.vulkan(vk_params)!
	defer { w_vk.release() }
	mut in_vk := input.vulkan(vk_params)!
	defer { in_vk.release() }

	mut d_in_vk := g_vk.gemm(w_vk)!
	defer { d_in_vk.release() }
	mut d_w_vk := g_vk.t()!.gemm(in_vk)!
	defer { d_w_vk.release() }

	d_in := d_in_vk.cpu()!
	d_weight := d_w_vk.cpu()!
	mut result := [d_in, d_weight, grad]
	if bias_needs_grad {
		batch_size := grad.shape[0]
		ones := vtl.ones[f32]([1, batch_size])
		result[2] = la.matmul[f32](ones, grad)!
	} else {
		result[2] = vtl.zeros[f32]([1, weight.shape[0]])
	}
	return result
}
