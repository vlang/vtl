module models

import vtl.autograd

// forward_vulkan_inference falls back to the ordinary CPU-capable path without
// a Vulkan-enabled build.
pub fn (mut nn Sequential[f32]) forward_vulkan_inference(input &autograd.Variable[f32]) !&autograd.Variable[f32] {
	validate_vulkan_inference[f32](nn, input)!
	return nn.forward(input)
}
