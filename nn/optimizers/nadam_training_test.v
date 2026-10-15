module optimizers

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.loss
import rand

fn train_xor_with_nadam() !(f64, f64, int) {
	rand.seed([u32(42), u32(0)])
	c := autograd.ctx[f64]()
	first := layers.linear_layer[f64](c, 2, 4)
	activation := layers.tanh_layer[f64](c, [4])
	second := layers.linear_layer[f64](c, 4, 1)
	criterion := loss.sigmoid_cross_entropy_loss[f64]()
	mut optimizer := nadam_optimizer[f64](learning_rate: 0.01)
	optimizer.build_params([first, second])
	mut x := c.variable(vtl.from_array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])!)
	y := vtl.from_array([0.0, 1.0, 1.0, 0.0], [4])!
	mut initial_loss := 0.0
	for epoch in 0 .. 5000 {
		hidden := first.forward(x)!
		activated := activation.forward(hidden)!
		prediction := second.forward(activated)!
		mut loss_value := criterion.loss(prediction, y)!
		if epoch == 0 { initial_loss = loss_value.value.get_nth(0) }
		loss_value.backprop()!
		optimizer.update()!
		x.grad = vtl.zeros_like[f64](x.value)
	}
	hidden := first.forward(x)!
	activated := activation.forward(hidden)!
	prediction := second.forward(activated)!
	mut correct := 0
	for i in 0 .. 4 {
		if (prediction.value.get_nth(i) > 0.0) == (y.get_nth(i) == 1.0) { correct++ }
	}
	final_loss := criterion.loss(prediction, y)!.value.get_nth(0)
	return initial_loss, final_loss, correct
}

fn test_nadam_converges_on_deterministic_xor() ! {
	initial, final_loss, correct := train_xor_with_nadam()!
	assert correct == 4, 'NAdam should classify all XOR cases, got ${correct}/4'
	assert final_loss < initial, 'NAdam loss did not decrease: ${initial} -> ${final_loss}'
}
