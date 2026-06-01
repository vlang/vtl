module layers

import vtl

// GPU ReLU/Sigmoid forward disabled until V 0.5.1 C codegen for VulkanTensor ops is stable.
// Linear/Conv2D Vulkan paths remain enabled. CPU activations via internal.* in relu_forward_f32.
pub fn relu_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	_ = x
	return none
}

pub fn sigmoid_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	_ = x
	return none
}
