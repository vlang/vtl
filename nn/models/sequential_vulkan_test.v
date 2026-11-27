module models

import os
import vtl
import vtl.autograd
import vtl.nn.internal

fn test_sequential_vulkan_inference_uses_standard_path_when_disabled() ! {
	if os.getenv('VTL_USE_VULKAN') == '1' {
		return
	}
	mut ctx := autograd.ctx[f32]()
	mut model := sequential_from_ctx[f32](ctx)
	model.input([3])
	model.relu()
	model.sigmoid()
	model.eval()
	ctx.no_grad = true

	input := vtl.from_array([f32(-1), 0.5, 2], [3])!
	variable := ctx.variable(input, requires_grad: false)
	actual := model.forward_vulkan_inference(variable)!
	expected := internal.sigmoid[f32](internal.relu[f32](input))
	actual_values := actual.value.to_array()
	expected_values := expected.to_array()
	assert actual_values.len == expected_values.len
	for i, want in expected_values {
		assert (actual_values[i] - want) * (actual_values[i] - want) < 1e-8
	}
	assert ctx.len() == 0
}
