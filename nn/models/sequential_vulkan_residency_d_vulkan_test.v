module models

import math
import os
import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.layers

fn test_sequential_vulkan_resident_activations_match_cpu_and_reuse_safely() ! {
	if os.getenv('VTL_TEST_VULKAN') != '1' || !layers.vulkan_linear_enabled() {
		return
	}
	mut ctx := autograd.ctx[f32]()
	mut model := sequential_from_ctx[f32](ctx)
	model.input([4])
	model.relu()
	model.sigmoid()
	model.eval()
	ctx.no_grad = true

	first := vtl.from_array([f32(-2), -0.5, 0.25, 3], [4])!
	first_var := ctx.variable(first, requires_grad: false)
	first_actual := model.forward_vulkan_inference(first_var)!
	first_expected := internal.sigmoid[f32](internal.relu[f32](first))
	assert_close_vulkan_resident(first_actual.value, first_expected)
	assert ctx.len() == 0

	second := vtl.from_array([f32(4), -3, 0, 0.75], [4])!
	second_var := ctx.variable(second, requires_grad: false)
	second_actual := model.forward_vulkan_inference(second_var)!
	second_expected := internal.sigmoid[f32](internal.relu[f32](second))
	assert_close_vulkan_resident(second_actual.value, second_expected)
	assert ctx.len() == 0
}

fn test_sequential_vulkan_resident_falls_back_across_unsupported_layer() ! {
	if os.getenv('VTL_TEST_VULKAN') != '1' || !layers.vulkan_linear_enabled() {
		return
	}
	mut ctx := autograd.ctx[f32]()
	mut model := sequential_from_ctx[f32](ctx)
	model.input([3])
	model.relu()
	model.tanh()
	model.sigmoid()
	model.eval()
	ctx.no_grad = true

	input := vtl.from_array([f32(-2), 0.5, 3], [3])!
	variable := ctx.variable(input, requires_grad: false)
	actual := model.forward_vulkan_inference(variable)!
	expected := internal.sigmoid[f32](internal.tanh[f32](internal.relu[f32](input)))
	assert_close_vulkan_resident(actual.value, expected)
	assert ctx.len() == 0
}

fn test_sequential_vulkan_resident_rejects_training_or_grad_tracking() ! {
	mut ctx := autograd.ctx[f32]()
	mut model := sequential_from_ctx[f32](ctx)
	model.input([1])
	model.relu()
	input := ctx.variable(vtl.ones[f32]([1]))
	assert_vulkan_inference_rejected(mut model, input, 'requires eval mode')
	model.eval()
	assert_vulkan_inference_rejected(mut model, input, 'requires no_grad')
	ctx.no_grad = true
	assert_vulkan_inference_rejected(mut model, input, 'untracked input')
}

fn assert_close_vulkan_resident(actual &vtl.Tensor[f32], expected &vtl.Tensor[f32]) {
	assert actual.shape == expected.shape
	actual_values := actual.to_array()
	expected_values := expected.to_array()
	for i, want in expected_values {
		assert math.abs(actual_values[i] - want) < 1e-4, 'resident Vulkan result mismatch at ${i}'
	}
}

fn assert_vulkan_inference_rejected(mut model Sequential[f32], input &autograd.Variable[f32], expected string) {
	model.forward_vulkan_inference(input) or {
		assert err.msg().contains(expected)
		return
	}
	assert false, 'forward_vulkan_inference should reject this call'
}
