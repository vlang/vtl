module layers

import storage
import vtl

// linear_forward_vulkan computes y = x·Wᵀ + b using Vulkan GEMM, then adds bias on CPU
// (broadcasting matches CPU linear_layer: la.matmul(x, W.t()) + b).
//
// Shapes: x [batch, in_features], weights [out_features, in_features], bias [1, out_features].
pub fn linear_forward_vulkan[T](x &vtl.Tensor[T], weights &vtl.Tensor[T], bias &vtl.Tensor[T], params storage.VulkanStorageParams) !&vtl.Tensor[T] {
	if x.rank() != 2 || weights.rank() != 2 || bias.rank() != 2 {
		return error('layers.linear_forward_vulkan: x, weights, and bias must be 2D')
	}
	batch := x.shape[0]
	in_f := x.shape[1]
	out_f := weights.shape[0]
	win := weights.shape[1]
	if in_f != win {
		return error('layers.linear_forward_vulkan: inner dims mismatch x(${batch}x${in_f}) vs W(${out_f}x${win})')
	}
	if bias.shape[0] != 1 || bias.shape[1] != out_f {
		return error('layers.linear_forward_vulkan: bias must be [1, ${out_f}], got ${bias.shape}')
	}
	wt := weights.t()!
	vk_x := x.vulkan(params)!
	defer {
		vk_x.release() or {}
	}
	vk_wt := wt.vulkan(params)!
	defer {
		vk_wt.release() or {}
	}
	vk_y := vk_x.gemm(vk_wt)!
	defer {
		vk_y.release() or {}
	}
	cpu_y := vk_y.cpu()!
	return cpu_y.add(bias)!
}

// relu_forward_vulkan applies ReLU on GPU and returns a CPU tensor.
pub fn relu_forward_vulkan[T](x &vtl.Tensor[T], params storage.VulkanStorageParams) !&vtl.Tensor[T] {
	vk := x.vulkan(params)!
	defer {
		vk.release() or {}
	}
	out := vk.relu()!
	defer {
		out.release() or {}
	}
	return out.cpu()!
}

// sigmoid_forward_vulkan applies sigmoid on GPU and returns a CPU tensor.
pub fn sigmoid_forward_vulkan[T](x &vtl.Tensor[T], params storage.VulkanStorageParams) !&vtl.Tensor[T] {
	vk := x.vulkan(params)!
	defer {
		vk.release() or {}
	}
	out := vk.sigmoid()!
	defer {
		out.release() or {}
	}
	return out.cpu()!
}