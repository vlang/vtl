module main

// nn_regression_sine: Train a small MLP to approximate f(x) = sin(x)
// over the range [-π, π]. Demonstrates regression with MSE loss.
//
// Network: input(1) → linear(16) → relu → linear(1) → mse_loss
// The model learns to map a scalar x to sin(x) without any explicit
// knowledge of the sine function — only from (x, y) training pairs.
import math
import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

// Training hyperparameters
const n_samples = 100
const epochs = 60
const learning_rate = 0.001

fn main() {
	ctx := autograd.ctx[f64]()

	// === Section 1: Generate training data ===
	// Sample n_samples points uniformly in [-π, π] and compute sin(x).
	// x_tensor is [n_samples, 1]: one feature per sample (a scalar x value).
	// y_tensor is [n_samples, 1]: matching target shape for MSE loss.
	mut x_data := []f64{len: n_samples}
	mut y_data := []f64{len: n_samples}

	for i in 0 .. n_samples {
		// Map index i to the range [-π, π]
		xi := -math.pi + (2.0 * math.pi * f64(i)) / f64(n_samples - 1)
		x_data[i] = xi
		y_data[i] = math.sin(xi)
	}

	x_tensor := vtl.from_array(x_data, [n_samples, 1])!
	// Target shape must match model output shape [n_samples, 1] for MSE
	y_tensor := vtl.from_array(y_data, [n_samples, 1])!

	mut x := ctx.variable(x_tensor, requires_grad: true)

	// === Section 2: Define the model ===
	// A two-layer MLP: 1 → 16 hidden units (ReLU) → 1 output.
	// MSE loss measures the mean squared error between prediction and target.
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([1])
	model.linear(16)
	model.relu()
	model.linear(1)
	model.mse_loss()

	// === Section 3: Set up the optimizer ===
	// SGD with a conservative learning rate to avoid gradient explosion.
	// build_params wires the optimizer to the model's learnable weight tensors.
	mut optimizer := optimizers.sgd[f64](learning_rate: learning_rate)
	optimizer.build_params(model.info.layers)

	// === Section 4: Training loop ===
	// Each epoch runs a full forward pass, computes MSE loss,
	// backpropagates gradients, and updates weights.
	println('Training MLP to approximate sin(x)...')
	println('Epoch | Loss')
	println('------|----------')

	for epoch in 0 .. epochs {
		y_pred := model.forward(x)!
		mut loss := model.loss(y_pred, y_tensor)!

		loss_val := loss.value.get([0])
		if epoch % 10 == 0 || epoch == epochs - 1 {
			println('  ${epoch:3d} | ${loss_val:.6f}')
		}

		loss.backprop()!
		optimizer.update()!
	}

	// === Section 5: Spot-check predictions ===
	// Evaluate the trained model at a few key points.
	// We reuse the same context (ctx) so the model weights are shared.
	println('')
	println('Spot-check after training:')
	println('  x       | sin(x)   | predicted')
	println('----------|----------|----------')

	check_points := [-math.pi / 2.0, 0.0, math.pi / 2.0, math.pi]
	for xv in check_points {
		x_check := vtl.from_array([xv], [1, 1])!
		mut xv_var := ctx.variable(x_check, requires_grad: false)
		pred := model.forward(xv_var)!
		predicted := pred.value.get([0, 0])
		expected := math.sin(xv)
		println('  ${xv:7.4f} | ${expected:8.4f} | ${predicted:8.4f}')
	}
}
