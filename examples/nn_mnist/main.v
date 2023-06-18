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
ctx := autograd.ctx[f64]()

// We create a neural network
mut model := models.sequential_from_ctx[f64](ctx)
model.input([1, 28, 28])
model.mse_loss()

// Load the MNIST dataset
mnist := datasets.load_mnist()!

// We reshape the data to fit the network
features := mnist.train_features.divide_scalar(u8(255))!.reshape([-1, 1, 28, 28])!
labels := mnist.train_labels

mut losses := []&vtl.Tensor[f64]{cap: epochs}

// Stochastic Gradient Descent
mut optimizer := optimizers.sgd[f64](learning_rate: 0.01)

println('Training...')

for epoch in 0 .. epochs {
	println('Epoch: ${epoch}')

	for batch_id in 0 .. batches {
		println('Batch id: ${batch_id}')

		offset := batch_id * batch_size

		mut x := ctx.variable(features.slice([offset, offset + batch_size])!)
		target := labels.slice([offset, offset + batch_size])!

		// Running input through the network
		y_pred := model.forward(mut x)!

		// Compute the loss
		mut loss := model.loss(y_pred, target)!

		println('Epoch: ${epoch}, Batch id: ${batch_id}, Loss: ${loss.value}')

		losses << loss.value

		// Compute the gradient (i.e. contribution of each parameter to the loss)
		loss.backprop()!

		// Correct the weights now that we have the gradient information
		optimizer.update()!
	}
}
