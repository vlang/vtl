module main

// nn_regression_sine_plot: Train an MLP to approximate sin(x) and visualize
// the results using vsl.plot. Demonstrates the full ML workflow:
// data generation → model definition → training loop → evaluation → plotting.
//
// Produces a plot comparing the true sin(x) curve with the model's predictions,
// plus a loss curve showing training convergence.
import math
import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers
import vsl.plot

// Training hyperparameters
const n_samples = 200
const epochs = 200
const learning_rate = 0.005

fn main() {
	ctx := autograd.ctx[f64]()

	// === Generate training data: sample sin(x) over [-π, π] ===
	mut x_data := []f64{len: n_samples}
	mut y_data := []f64{len: n_samples}

	for i in 0 .. n_samples {
		xi := -math.pi + (2.0 * math.pi * f64(i)) / f64(n_samples - 1)
		x_data[i] = xi
		y_data[i] = math.sin(xi)
	}

	x_tensor := vtl.from_array(x_data, [n_samples, 1])!
	y_tensor := vtl.from_array(y_data, [n_samples, 1])!

	mut x := ctx.variable(x_tensor, requires_grad: true)

	// === Define the model ===
	// Two hidden layers with ReLU: 1 → 32 → 32 → 1
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([1])
	model.linear(32)
	model.relu()
	model.linear(32)
	model.relu()
	model.linear(1)
	model.mse_loss()

	// === Optimizer ===
	mut optimizer := optimizers.sgd[f64](learning_rate: learning_rate)
	optimizer.build_params(model.info.layers)

	// === Training loop ===
	println('Training MLP to approximate sin(x)...')
	mut losses := []f64{cap: epochs}

	for epoch in 0 .. epochs {
		y_pred := model.forward(x)!
		mut loss := model.loss(y_pred, y_tensor)!
		loss_val := loss.value.get([0])
		losses << loss_val

		if epoch % 50 == 0 || epoch == epochs - 1 {
			println('  Epoch ${epoch:3d} | Loss: ${loss_val:.6f}')
		}

		loss.backprop()!
		optimizer.update()!
	}

	// === Evaluate the trained model ===
	println('\nGenerating predictions for plotting...')
	mut y_predicted := []f64{len: n_samples}

	for i in 0 .. n_samples {
		x_check := vtl.from_array([x_data[i]], [1, 1])!
		mut xv := ctx.variable(x_check, requires_grad: false)
		pred := model.forward(xv)!
		y_predicted[i] = pred.value.get([0, 0])
	}

	// === Plot 1: True vs Predicted curves ===
	mut plt := plot.Plot.new()
	plt.scatter(
		x:    x_data
		y:    y_data
		mode: 'lines'
		line: plot.Line{
			color: '#2196F3'
			width: 2.0
		}
		name: 'sin(x) — true'
	)
	plt.scatter(
		x:    x_data
		y:    y_predicted
		mode: 'lines'
		line: plot.Line{
			color: '#FF5722'
			width: 2.0
			dash:  'dash'
		}
		name: 'MLP prediction'
	)
	plt.layout(
		title: 'VTL Neural Network: Learning sin(x)'
		xaxis: plot.Axis{
			title: plot.AxisTitle{
				text: 'x'
			}
		}
		yaxis: plot.Axis{
			title: plot.AxisTitle{
				text: 'y'
			}
		}
	)
	plt.show()!

	// === Plot 2: Training loss curve ===
	mut loss_plt := plot.Plot.new()
	epoch_x := []f64{len: epochs, init: f64(index)}
	loss_plt.scatter(
		x:    epoch_x
		y:    losses
		mode: 'lines'
		line: plot.Line{
			color: '#4CAF50'
			width: 2.0
		}
		name: 'MSE Loss'
	)
	loss_plt.layout(
		title: 'Training Loss Convergence'
		xaxis: plot.Axis{
			title: plot.AxisTitle{
				text: 'Epoch'
			}
		}
		yaxis: plot.Axis{
			title: plot.AxisTitle{
				text: 'Loss'
			}
		}
	)
	loss_plt.show()!

	println('\nDone! Two plots opened in browser.')
}
