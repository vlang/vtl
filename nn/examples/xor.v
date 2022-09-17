module main

import vtl
import vtl.autograd
import vtl.nn

fn main() {
	ctx := autograd.new_ctx<f64>()

	bsz := 32

	x_train_bool := vtl.random(0, 2, [bsz * 100, 2])

	mut x_train := ctx.variable(x_train_bool)
	y_bool := x_train_bool.slice_hilo([]int{}, [0])?.equal(x_train_bool.slice_hilo([]int{},
		[1])?)?
	y := y_bool

	mut net := nn.new_nn<f64>(ctx)
	net.input([2])
	net.linear(3)
	net.relu()
	net.linear(1)
	net.sgd(learning_rate: 0.7)
	net.sigmoid_cross_entropy_loss()

	epochs := 50
	batches := 100

	mut losses := []&vtl.Tensor<f64>{cap: epochs * batches}

	for epoch in 0 .. epochs {
		for batch_id in 0 .. batches {
			offset := batch_id * bsz

			mut x := x_train.slice([offset, offset + bsz])?
			target := y.slice([offset, offset + bsz])?

			mut loss := net.forward(mut x)?
			losses << loss.value

			loss.backprop()?
		}
	}
}
