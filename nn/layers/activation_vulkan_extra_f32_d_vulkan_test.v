module layers

import math
import os
import vtl
import vtl.autograd
import vtl.nn.internal

fn test_extra_activation_layers_vulkan_f32_match_cpu() ! {
	if os.getenv('VTL_USE_VULKAN') != '1' {
		return
	}
	values := [f32(-20), -3, -1, 0, 1, 3, 20, f32(math.inf(-1))]

	softplus_context := autograd.ctx[f32]()
	softplus_input := softplus_context.variable(vtl.from_array(values, [values.len])!)
	mut softplus_gpu := softplus_layer[f32](softplus_context, [values.len]).forward(softplus_input)!
	softplus_cpu := internal.softplus[f32](softplus_input.value)
	softplus_gpu.backprop()!

	selu_context := autograd.ctx[f32]()
	selu_input := selu_context.variable(vtl.from_array(values, [values.len])!)
	mut selu_gpu := selu_layer[f32](selu_context, [values.len]).forward(selu_input)!
	selu_cpu := internal.selu[f32](selu_input.value)
	selu_gpu.backprop()!

	hardswish_context := autograd.ctx[f32]()
	hardswish_input := hardswish_context.variable(vtl.from_array(values, [values.len])!)
	mut hardswish_gpu := hardswish_layer[f32](hardswish_context, [values.len]).forward(hardswish_input)!
	hardswish_cpu := internal.hardswish[f32](hardswish_input.value)
	hardswish_gpu.backprop()!

	for i in 0 .. values.len {
		assert math.abs(f64(softplus_gpu.value.get_nth(i) - softplus_cpu.get_nth(i))) < 1e-5
		assert math.abs(f64(selu_gpu.value.get_nth(i) - selu_cpu.get_nth(i))) < 1e-5
		assert math.abs(f64(hardswish_gpu.value.get_nth(i) - hardswish_cpu.get_nth(i))) < 1e-5

		x := f64(values[i])
		softplus_grad := if x >= 0 {
			1.0 / (1.0 + math.exp(-x))
		} else {
			exp_x := math.exp(x)
			exp_x / (1.0 + exp_x)
		}
		selu_grad := if x > 0 {
			1.0507009873554805
		} else {
			1.0507009873554805 * 1.6732632423543772 * math.exp(x)
		}
		hardswish_grad := if x <= -3.0 {
			0.0
		} else if x >= 3.0 {
			1.0
		} else {
			(2.0 * x + 3.0) / 6.0
		}
		assert math.abs(f64(softplus_input.grad.get_nth(i)) - softplus_grad) < 1e-5
		assert math.abs(f64(selu_input.grad.get_nth(i)) - selu_grad) < 1e-5
		assert math.abs(f64(hardswish_input.grad.get_nth(i)) - hardswish_grad) < 1e-5
	}
}
