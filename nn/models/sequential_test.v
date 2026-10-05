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
