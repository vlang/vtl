module models

import vtl
import vtl.autograd
import vtl.nn.types

fn test_nnc() {
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.sigmoid()
	assert nn.info.layers.len == 2
	assert nn.info.layers[0].output_shape() == [1, 2]
	assert nn.info.layers[1].output_shape() == [1, 2]
}

fn test_nn() {
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
}

fn test_new_sequential_losses() ! {
	c := autograd.ctx[f64]()
	pred := c.variable(vtl.from_array([1.0, -0.5], [2])!)
	target := vtl.from_array([0.0, -1.0], [2])!
	hinge_target := vtl.from_array([1.0, -1.0], [2])!
	focal_target := vtl.from_array([0.0, 1.0], [2])!
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})

	nn.l1_loss()
	mut l1 := nn.loss(pred, target)!
	assert l1.value.shape == [1]
	l1.backprop()!

	nn.hinge_loss()
	mut hinge := nn.loss(pred, hinge_target)!
	assert hinge.value.shape == [1]
	hinge.backprop()!

	nn.focal_loss()
	mut focal := nn.loss(pred, focal_target)!
	assert focal.value.shape == [1]
	focal.backprop()!
}

fn test_added_activation_layers_in_sequential() {
	mut nn := sequential[f64]()
	nn.input([3])
	nn.softplus()
	nn.selu()
	nn.hardswish()
	assert nn.info.layers.len == 4
	assert nn.info.layers[1].output_shape() == [3]
	assert nn.info.layers[2].output_shape() == [3]
	assert nn.info.layers[3].output_shape() == [3]
}

fn test_sequential_no_grad_does_not_record_inference_graph() ! {
	mut ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([2])
	nn.linear(1)
	nn.eval()

	input := ctx.variable(vtl.from_array([1.0, 2.0], [1, 2])!)
	ctx.no_grad = true
	output := nn.forward(input)!
	assert ctx.len() == 0
	assert !output.is_grad_needed()

	ctx.no_grad = false
	assert !output.is_grad_needed()

	mut prediction := output.sum()!
	assert ctx.len() == 0
	assert !prediction.is_grad_needed()
	prediction.backprop()!
	assert input.grad.to_array() == [0.0, 0.0]
	for layer in nn.info.layers {
		for parameter in layer.variables() {
			assert parameter.grad.to_array().all(it == 0.0)
		}
	}
}
