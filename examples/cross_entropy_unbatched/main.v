module main

import math
import vtl
import vtl.autograd
import vtl.nn.loss

fn main() {
	ctx := autograd.ctx[f64]()
	mut logits := ctx.variable(vtl.from_array([0.0, math.log(3.0)], [2])!)
	target := vtl.from_array([1], [])!
	objective := loss.cross_entropy_loss_indices[f64](unsafe { nil }, reduction: 'mean')
	mut loss_value := objective.loss(logits, target)!
	println('cross-entropy: ${loss_value.value.get_nth(0):.6f}')
	loss_value.backprop()!
	println('gradient: ${logits.grad.to_array()}')
}
