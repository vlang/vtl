module main

import vtl
import vtl.autograd
import vtl.datasets
import vtl.nn.models
import vtl.nn.optimizers

const (
	batch_size = 32
	epochs     = 1
	batches    = 100
)

// Autograd context / neuralnet graph
ctx := autograd.ctx<f64>()

// We create a neural network
mut model := models.sequential_from_ctx<f64>(ctx)
model.input([1, 28, 28])
model.mse_loss()

mut train_ds := datasets.load_mnist(.train, batch_size: batch_size)!

mut losses := []&vtl.Tensor<f64>{cap: epochs}

// Stochastic Gradient Descent
mut optimizer := optimizers.sgd<f64>(learning_rate: 0.01)

for epoch in 0 .. epochs {
	mut batch_id := 0
	for {
		batch := train_ds.next() or { break }

		xt := batch.features.divide_scalar(255.0)?.reshape([-1.0, 1.0, 28.0, 28.0])?
		mut x := ctx.variable(xt)
		target := batch.labels

		// Running input through the network
		y_pred := model.forward(mut x)?

		// Compute the loss
		mut loss := model.loss(y_pred, target)?

		println('Epoch: $epoch, Batch id: $batch_id, Loss: $loss.value')

		losses << loss.value

		// Compute the gradient (i.e. contribution of each parameter to the loss)
		loss.backprop()?

		// Correct the weights now that we have the gradient information
		optimizer.update()?

		batch_id++
	}
}
