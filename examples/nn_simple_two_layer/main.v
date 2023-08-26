module main

import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

const (
	batch_size = 64
	input_dim  = 1000
	hidden_dim = 100
	output_dim = 10
	epochs     = 500
)

// Learning XOR function with a neural network.

// Autograd context / neuralnet graph
ctx := autograd.ctx[f64]()

// Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x_data := vtl.random(0.0, 1.0, [batch_size, input_dim])
y := vtl.random(0.0, 1.0, [batch_size, output_dim])

// We need to convert the bool tensor to a float tensor
mut x := ctx.variable(x_data,
	requires_grad: true
)

// We create a simple two layer neural network
mut model := models.sequential_from_ctx[f64](ctx)
model.input([input_dim])
model.linear(hidden_dim)
model.linear(output_dim)
model.relu()
model.mse_loss()

// Stochastic Gradient Descent
mut optimizer := optimizers.sgd[f64](learning_rate: 1.0e-4)

mut losses := []&vtl.Tensor[f64]{cap: epochs}

// Learning loop
for epoch in 0 .. epochs {
	println('Epoch: ${epoch}')

	// Running input through the network
	y_pred := model.forward(mut x)!

	// Compute the loss
	mut loss := model.loss(y_pred, y)!

	println('Epoch: ${epoch}, Loss: ${loss.value}')

	losses << loss.value

	// Compute the gradient (i.e. contribution of each parameter to the loss)
	loss.backprop()!

	// Correct the weights now that we have the gradient information
	optimizer.update()!
}
