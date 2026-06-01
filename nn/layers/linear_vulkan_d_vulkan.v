module layers

import vtl.storage
import vtl.la
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers
import vtl.nn.types
import vsl.vulkan

// linear_forward_vulkan_f32 is the Sequential f32 entry point (opt-in via VTL_USE_VULKAN=1).
pub fn linear_forward_vulkan_f32(x &vtl.Tensor[f32], weights &vtl.Tensor[f32], bias &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	if !vulkan_linear_enabled() {
		return error('linear_forward_vulkan_f32: set VTL_USE_VULKAN=1 to enable')
	}
	return linear_forward_vulkan(x, weights, bias)
}

// linear_forward_vulkan computes y = x * W^T + b using Vulkan GEMM
// x: [M, K] (input matrix)
// W: [N, K] (weights matrix)
// b: [N] or [1, N] (bias vector)
// Returns: [M, N] (output matrix)
pub fn linear_forward_vulkan[T](x &Tensor[T], weights &Tensor[T], bias &Tensor[T]) !&Tensor[T] {
	assert x.is_matrix()
	assert weights.is_matrix()
	assert bias.is_vector() || (bias.is_matrix() && bias.shape[0] == 1)

	// Convert to Vulkan tensors for GPU computation
	mut x_vk := x.vulkan()!
	mut w_vk := weights.vulkan()!
	mut b_vk := bias.vulkan()!

	// GEMM: x * W^T
	mut gemm_result := x_vk.gemm(w_vk.t())!

	// Add bias (broadcast if needed)
	mut biased := gemm_result.vulkan()!
	if b_vk.shape[0] == 1 && gemm_result.shape[0] > 1 {
		// Broadcast bias to all rows
		mut result_data := biased.data.device.buffer(f32, biased.size)!
		mut biased_arr := biased.to_array()!
		mut b_val := biased_arr[0]
		for i in 0 .. biased.shape[0] {
			for j in 0 .. biased.shape[1] {
				result_data.data[i * biased.shape[1] + j] = biased_arr[j] + b_val
			}
		}
		mut result := &Tensor[T]{
			data:    result_data
			memory:  .row_major
			size:    result_data.size
			shape:   biased.shape
			strides: [biased.shape[1], 1]
		}
		x_vk.release()
		w_vk.release()
		b_vk.release()
		gemm_result.release()
		biased.release()
		result_data.release()
		return result
	} else {
		// Element-wise add
		mut result := biased.vulkan()!
		mut b_arr := b_vk.to_array()!
		mut res_arr := result.to_array()!
		for i in 0 .. res_arr.len {
			res_arr[i] += b_arr[i % b_arr.len]
		}
		mut result_tensor := &Tensor[T]{
			data:    result.data
			memory:  result.memory
			size:    result.size
			shape:   result.shape
			strides: result.strides
		}
		x_vk.release()
		w_vk.release()
		b_vk.release()
		gemm_result.release()
		biased.release()
		result.release()
		return result_tensor
	}
}

// relu_forward_vulkan applies ReLU activation: max(0, x)
pub fn relu_forward_vulkan[T](x &Tensor[T]) !&Tensor[T] {
	mut x_vk := x.vulkan()!
	mut result := x_vk.relu()!
	x_vk.release()
	return result
}

// sigmoid_forward_vulkan applies Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid_forward_vulkan[T](x &Tensor[T]) !&Tensor[T] {
	mut x_vk := x.vulkan()!
	mut result := x_vk.sigmoid()!
	x_vk.release()
	return result
}
