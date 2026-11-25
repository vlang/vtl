module main

import os
import rand
import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

fn main() {
	rand.seed([u32(42), u32(0)])
	train_ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](train_ctx)
	model.input([2])
	model.linear(1)
	model.mse_loss()

	mut train_optimizer := optimizers.sgd[f64](learning_rate: 0.01)
	train_optimizer.build_params(model.info.layers)
	train_input := train_ctx.variable(vtl.from_array[f64]([1.0, 0.0, 0.0, 1.0], [2, 2])!)
	target := vtl.from_array[f64]([1.0, -1.0], [2, 1])!
	prediction := model.forward(train_input)!
	mut loss := model.loss(prediction, target)!
	loss.backprop()!
	train_optimizer.update()!

	checkpoint_path := 'sequential-inference-checkpoint.json'
	defer {
		os.rm(checkpoint_path) or {}
	}
	model.save_inference_checkpoint(checkpoint_path)!

	mut inference_ctx := autograd.ctx[f64]()
	mut loaded := models.sequential_from_ctx[f64](inference_ctx)
	loaded.input([2])
	loaded.linear(1)
	loaded.load_inference_checkpoint(checkpoint_path)!
	loaded.eval()
	inference_ctx.no_grad = true
	input := inference_ctx.variable(vtl.from_array[f64]([1.0, 0.0], [1, 2])!,
		requires_grad: false
	)
	output := loaded.forward(input)!
	println('prediction: ${output.value}')
}
