module models

import vtl.autograd

fn validate_vulkan_inference[T](nn &Sequential[T], input &autograd.Variable[T]) ! {
	if nn.training {
		return error('forward_vulkan_inference requires eval mode')
	}
	if !input.context.no_grad || input.requires_grad {
		return error('forward_vulkan_inference requires no_grad and an untracked input')
	}
}

fn is_vulkan_resident_activation(layer_type string) bool {
	return layer_type in ['ReLULayer', 'SigmoidLayer']
}
