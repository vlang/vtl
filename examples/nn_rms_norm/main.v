module main

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models

fn main() {
	ctx := autograd.ctx[f32]()
	mut model := models.sequential_from_ctx[f32](ctx)
	model.input([4])
	model.rms_norm([4], layers.RMSNormConfig{})
	input := ctx.variable(vtl.from_array([f32(1), 2, 3, 4, 5, 6, 7, 8], [2, 4])!)
	output := model.forward(input)!
	println('Input shape: ${input.value.shape}')
	println('Output shape: ${output.value.shape}')
	println('RMSNorm trainable tensors: ${model.info.layers[1].variables().len}')
}
