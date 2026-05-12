module main

import vtl
import vtl.autograd
import vtl.datasets
import vtl.nn.models
import vtl.nn.optimizers
import vtl.runtime

const batch_size = 32
const epochs = 3
const batches = 100

// A simple feedforward network trained on MNIST.
// Architecture: flatten(784) → linear(128) → relu → linear(64) → relu → linear(10) → mse_loss
// Labels are scalar class indices (0–9); the network predicts a score for each class.

fn main() {
	// Autograd context / neuralnet graph
	mut ctx := autograd.ctx[f64]()
	policy := runtime.policy_from_env() or { runtime.ExecutionPolicy{} }
	runtime.apply_policy[f64](mut ctx, policy)
	println('Runtime backend policy: backend=${policy.backend}, strict=${policy.strict}')

	// Load the MNIST dataset (downloads automatically on first run)
	println('Loading MNIST dataset...')
	mnist := datasets.load_mnist()!

	// Normalize pixel values from [0, 255] to [0.0, 1.0] and flatten to [60000, 784]
	features := mnist.train_features.as_f64().divide_scalar(255.0)!.reshape([-1, 784])!

	// Labels as float, shape [60000] — each value is the class index 0–9
	labels := mnist.train_labels.as_f64()

	// Build the network
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([784])
	model.linear(128)
	model.relu()
	model.linear(64)
	model.relu()
	model.linear(10)
	model.mse_loss()

	// Stochastic Gradient Descent
	mut optimizer := optimizers.sgd[f64](learning_rate: 0.01)
	// Register model parameters with the optimizer
	optimizer.build_params(model.info.layers)

	println('Training...')

	for epoch in 0 .. epochs {
		println('Epoch: ${epoch}')

		for batch_id in 0 .. batches {
			offset := batch_id * batch_size

			mut x := ctx.variable(features.slice([offset, offset + batch_size])!,
				requires_grad: true
			)
			// One-hot encode labels to shape [batch_size, 10] to match model output
			label_slice := labels.slice([offset, offset + batch_size])!
			mut target := vtl.zeros[f64]([batch_size, 10], vtl.TensorData{})
			for i in 0 .. batch_size {
				class_idx := int(label_slice.get([i]))
				target.set([i, class_idx], 1.0)
			}

			// Forward pass
			y_pred := model.forward(x)!

			// Loss
			mut loss := model.loss(y_pred, target)!

			println('Epoch: ${epoch}, Batch: ${batch_id}, Loss: ${loss.value}')

			// Backpropagation
			loss.backprop()!

			// Weight update
			optimizer.update()!
		}
	}

	println('Done.')
}
