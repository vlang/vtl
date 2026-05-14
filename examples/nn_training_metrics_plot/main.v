module main

// nn_training_metrics_plot: Train a classifier on the XOR problem and
// visualize training metrics (loss, accuracy) using vsl.plot.
//
// This example shows:
//   1. Defining a model with VTL's Sequential API
//   2. Running a training loop with batch evaluation
//   3. Computing accuracy metrics during training
//   4. Plotting loss and accuracy curves with vsl.plot
import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers
import vsl.plot

const epochs = 300
const learning_rate = 0.1

fn main() {
	ctx := autograd.ctx[f64]()

	// === XOR dataset ===
	// Input: 4 samples of 2 features each
	x_data := [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
	y_data := [0.0, 1.0, 1.0, 0.0]

	x_tensor := vtl.from_array(x_data, [4, 2])!
	y_tensor := vtl.from_array(y_data, [4, 1])!

	mut x := ctx.variable(x_tensor, requires_grad: true)

	// === Model: 2 → 8 → 8 → 1 ===
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([2])
	model.linear(8)
	model.relu()
	model.linear(8)
	model.relu()
	model.linear(1)
	model.mse_loss()

	mut optimizer := optimizers.sgd[f64](learning_rate: learning_rate)
	optimizer.build_params(model.info.layers)

	// === Training loop with metrics collection ===
	println('Training XOR classifier...')
	mut losses := []f64{cap: epochs}
	mut accuracies := []f64{cap: epochs}

	for epoch in 0 .. epochs {
		y_pred := model.forward(x)!
		mut loss := model.loss(y_pred, y_tensor)!
		loss_val := loss.value.get([0])
		losses << loss_val

		// Compute accuracy: threshold at 0.5
		mut correct := 0
		for i in 0 .. 4 {
			predicted := y_pred.value.get([i, 0])
			target := y_data[i]
			pred_class := if predicted >= 0.5 { 1.0 } else { 0.0 }
			if pred_class == target {
				correct++
			}
		}
		accuracies << f64(correct) / 4.0

		if epoch % 50 == 0 || epoch == epochs - 1 {
			println('  Epoch ${epoch:3d} | Loss: ${loss_val:.6f} | Accuracy: ${f64(correct) / 4.0:.2f}')
		}

		loss.backprop()!
		optimizer.update()!
	}

	// === Plot: Loss and Accuracy ===
	mut plt := plot.Plot.new()
	epoch_x := []f64{len: epochs, init: f64(index)}

	plt.scatter(
		x:     epoch_x
		y:     losses
		mode:  'lines'
		line:  plot.Line{
			color: '#F44336'
			width: 2.0
		}
		name:  'Loss (MSE)'
		yaxis: 'y'
	)
	plt.scatter(
		x:     epoch_x
		y:     accuracies
		mode:  'lines'
		line:  plot.Line{
			color: '#2196F3'
			width: 2.0
		}
		name:  'Accuracy'
		yaxis: 'y2'
	)
	plt.layout(
		title: 'VTL Training Metrics: XOR Classification'
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
	plt.show()!

	// === Final predictions ===
	println('\nFinal predictions:')
	println('  Input     | Target | Predicted')
	println('  ----------|--------|----------')
	inputs := [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
	for i, input in inputs {
		x_check := vtl.from_array(input, [1, 2])!
		mut xv := ctx.variable(x_check, requires_grad: false)
		pred := model.forward(xv)!
		println('  [${input[0]:.0f}, ${input[1]:.0f}]   |  ${y_data[i]:.0f}     |  ${pred.value.get([
			0,
			0,
		]):.4f}')
	}
}
