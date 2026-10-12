module main

import vtl
import vtl.autograd
import vtl.nn.loss

fn main() {
	ctx := autograd.ctx[f64]()
	log_probs := vtl.from_array([-2.0, -0.5, -1.0], [3])!
	mut prediction := ctx.variable(log_probs)
	target := vtl.from_array([1], [])!
	weights := vtl.from_array([2.0, 4.0, 8.0], [3])!

	objective := loss.nll_loss_indices[f64](weights, reduction: 'mean')
	mut loss_value := objective.loss(prediction, target)!
	println('unbatched NLL: ${loss_value.value.get_nth(0):.3f}')
	loss_value.backprop()!
	println('gradient: ${prediction.grad.to_array()}')
}
