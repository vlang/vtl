module layers

import vtl
import vtl.storage

// linear_forward_vulkan_f32 is the Sequential f32 entry point (opt-in via VTL_USE_VULKAN=1).
pub fn linear_forward_vulkan_f32(x &vtl.Tensor[f32], weights &vtl.Tensor[f32], bias &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	if !vulkan_linear_enabled() {
		return error('linear_forward_vulkan_f32: set VTL_USE_VULKAN=1 to enable')
	}
	return linear_forward_vulkan(x, weights, bias)
}

// linear_forward_vulkan computes y = x * W^T + b using Vulkan GEMM; returns host CPU tensor.
pub fn linear_forward_vulkan[T](x &vtl.Tensor[T], weights &vtl.Tensor[T], bias &vtl.Tensor[T]) !&vtl.Tensor[T] {
	assert x.is_matrix()
	assert weights.is_matrix()
	assert bias.is_vector() || (bias.is_matrix() && bias.shape[0] == 1)

	mut x_vk := x.vulkan()!
	defer { x_vk.release() }
	dev := x_vk.data.data.device
	vk_params := storage.vulkan_params_for_device(dev)
	mut w_vk := weights.vulkan(vk_params)!
	defer { w_vk.release() }
	mut b_vk := bias.vulkan(vk_params)!
	defer { b_vk.release() }

	mut gemm_result := x_vk.gemm(w_vk.t()!)!
	defer { gemm_result.release() }

	if b_vk.shape[0] == 1 && gemm_result.shape[0] > 1 {
		mut biased_arr := gemm_result.data.to_array()!
		b_row := b_vk.data.to_array()!
		for i in 0 .. gemm_result.shape[0] {
			for j in 0 .. gemm_result.shape[1] {
				biased_arr[i * gemm_result.shape[1] + j] += b_row[j]
			}
		}
		return vtl.from_array(biased_arr, gemm_result.shape)!
	}
	mut res_arr := gemm_result.data.to_array()!
	b_arr := b_vk.data.to_array()!
	for i in 0 .. res_arr.len {
		res_arr[i] += b_arr[i % b_arr.len]
	}
	return vtl.from_array(res_arr, gemm_result.shape)!
}

// relu_forward_vulkan applies ReLU activation: max(0, x)
pub fn relu_forward_vulkan[T](x &vtl.Tensor[T]) !&vtl.Tensor[T] {
	mut x_vk := x.vulkan()!
	defer { x_vk.release() }
	x_vk.relu()!
	return x_vk.cpu()!
}

// sigmoid_forward_vulkan applies Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid_forward_vulkan[T](x &vtl.Tensor[T]) !&vtl.Tensor[T] {
	mut x_vk := x.vulkan()!
	defer { x_vk.release() }
	x_vk.sigmoid()!
	return x_vk.cpu()!
}
