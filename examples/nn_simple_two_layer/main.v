module main

import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

const batch_size = 64
const input_dim = 100
const hidden_dim = 32
const output_dim = 10
const epochs = 20

// A simple two-layer neural network trained with MSE loss.
// Architecture: input(100) → linear(32) → relu → linear(10) → mse_loss

fn main() {
	// Autograd context / neuralnet graph
	ctx := autograd.ctx[f64]()

	// Create random tensors to hold inputs and outputs
	x_data := vtl.random(0.0, 1.0, [batch_size, input_dim])
	y := vtl.random(0.0, 1.0, [batch_size, output_dim])

	// Wrap input in a Variable so gradients are tracked
	mut x := ctx.variable(x_data, requires_grad: true)

	// Build the network: input → linear(hidden) → relu → linear(output) → mse_loss
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([input_dim])
	model.linear(hidden_dim)
	model.relu()
	model.linear(output_dim)
	model.mse_loss()

	// Stochastic Gradient Descent
	mut optimizer := optimizers.sgd[f64](learning_rate: 1.0e-3)
	// Register model parameters with the optimizer
	optimizer.build_params(model.info.layers)

	mut losses := []&vtl.Tensor[f64]{cap: epochs}

	// Training loop
	for epoch in 0 .. epochs {
		// Running input through the network
		y_pred := model.forward(x)!

		// Compute the loss
		mut loss := model.loss(y_pred, y)!

		println('Epoch: ${epoch}, Loss: ${loss.value}')
		losses << loss.value

		// Backpropagate gradients
		loss.backprop()!

		// Update weights
		optimizer.update()!
	}
}
