module main

import vtl
import vtl.autograd
import vtl.nn

fn main() {
	ctx := autograd.new_ctx<f64>()

	bsz := 32

	x_train_bool := vtl.random(0, 2, [bsz * 100, 2])

	x_train := ctx.variable(x_train_bool)

	mut net := nn.new_nn<f64>(ctx)
	net.input([2])
	net.linear(3)
	net.relu()
	net.linear(1)
	net.sgd(learning_rate: 0.7)
	net.sigmoid_cross_entropy_loss()

	epochs := 1

	mut losses := []&vtl.Tensor<f64>{cap: epochs}

	for epoch in 0 .. epochs {
		mut i := 0
		for {
			i++
			if i == 10 {
				break
			}
		}
	}
}
