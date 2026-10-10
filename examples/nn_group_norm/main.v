module main

import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.layers

fn main() {
	ctx := autograd.ctx[f32]()
	mut model := models.sequential_from_ctx[f32](ctx)
	model.input([4, 8, 8])
	model.group_norm(2, layers.GroupNormConfig{})
	input := ctx.variable(vtl.ones[f32]([2, 4, 8, 8]))
	output := model.forward(input)!
	println('Input shape: ${input.value.shape}')
	println('Output shape: ${output.value.shape}')
	println('GroupNorm parameters: ${model.info.layers[1].variables().len}')
}
