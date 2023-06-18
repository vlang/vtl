module main

import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

const (
	batch_size = 32
	epochs     = 6
	batches    = 100
)

// Learning XOR function with a neural network.

// Autograd context / neuralnet graph
ctx := autograd.ctx[f64]()

// We will create a tensor of size 3200 (100 batches of size 32)
// We create it as int between [0, 2] and convert to bool
x_train_bool := vtl.random(0, 2, [batch_size * 100, 2]).as_bool()

// Let's build our truth labels. We need to apply xor between the 2 columns of the tensors
x_train_bool_1 := x_train_bool.slice([]int{}, [0])!
x_train_bool_2 := x_train_bool.slice_hilo([]int{}, [1])!

println(x_train_bool_1.shape)
println(x_train_bool_2.shape)

y_bool := x_train_bool_1.equal(x_train_bool_2)!

// We need to convert the bool tensor to a float tensor
mut x_train := ctx.variable(x_train_bool.as_f64(),
	requires_grad: true
)
y := y_bool.as_f64()

// We create a neural network with 2 inputs, 2 hidden layers of 4 neurons each and 1 output
// We use the sigmoid activation function
mut model := models.sequential_from_ctx[f64](ctx)
model.input([2])
model.linear(3)
model.relu()
model.linear(1)
model.sigmoid_cross_entropy_loss()

// Stochastic Gradient Descent
mut optimizer := optimizers.sgd[f64](learning_rate: 0.01)

mut losses := []&vtl.Tensor[f64]{cap: epochs * batches}

// Learning loop
for epoch in 0 .. epochs {
	println('Epoch: ${epoch}')
	for batch_id in 0 .. batches {
		// minibatch offset in the Tensor
		offset := batch_id * batch_size
		mut x := x_train.slice([offset, offset + batch_size])!
		target := y.slice([offset, offset + batch_size])!

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
