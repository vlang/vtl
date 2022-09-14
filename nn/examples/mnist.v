module main

import vtl
import vtl.autograd
import vtl.datasets
import vtl.nn

fn main() {
	ctx := autograd.new_ctx<f64>()
	mut net := nn.new_nn<f64>(ctx)
	net.input([1, 1, 28, 28])
	net.mse_loss()

	epochs := 1

	mut losses := []&vtl.Tensor<f64>{cap: epochs}

	for epoch in 0 .. epochs {
		mut train_ds := datasets.load_mnist(.train, batch_size: 6)?
		mut i := 0
		for {
			batch := train_ds.next() or { break }

			xt := batch.features.divide_scalar(u8(255))?.reshape([-1, 1, 28, 28])?
			mut x := ctx.variable(xt)
			target := batch.labels

			output := net.forward(mut x)?

			loss := net.loss(output, target)?
			print(loss)

			i++
			if i == 10 {
				break
			}
		}
	}
}
