module main

// Multiclass classification example using softmax cross-entropy loss.
//
// Inspired by Arraymancer ex02 (handwritten digits / MNIST):
//   https://github.com/mratsim/Arraymancer/blob/master/examples/ex02_handwritten_digits_recognition.nim
//
// Architecture:
//   Input (2) → Linear(8) → ReLU → Linear(3) → SoftmaxCrossEntropyLoss
//
// Dataset: 60 synthetic 2-D points arranged in 3 well-separated clusters.
// Points are interleaved (class 0, 1, 2, 0, 1, 2, …) so every mini-batch
// contains a balanced mix of all three classes — the same strategy used by
// Arraymancer's XOR and MNIST examples.
//
// Key hyperparameter notes (following Arraymancer conventions):
//   - learning_rate: 0.01  (safe for plain SGD + softmax CE)
//   - batch_size:     6    (each batch sees all 3 classes twice)
//   - epochs:       500    (small dataset benefits from many passes)
//   - rand seed:    42     (fixed for reproducibility, as in Arraymancer)

import math
import rand
import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

// ── Dataset constants ──────────────────────────────────────────────────────────
const n_classes = 3
const n_features = 2
const n_total = 60 // 20 samples per class, interleaved

// ── Training constants ─────────────────────────────────────────────────────────
const batch_size = 6
const epochs = 500
const learning_rate = 0.01

fn main() {
	// Fix the random seed for reproducible weight initialisation.
	// This mirrors Arraymancer's `randomize(42)` convention.
	rand.seed([u32(42), u32(0)])

	ctx := autograd.ctx[f64]()

	mut x_data := []f64{len: n_total * n_features}
	mut y_data := []f64{len: n_total * n_classes}

	// Interleave classes: row i belongs to class (i % n_classes).
	// Each batch of 6 therefore sees exactly 2 samples from every class.
	// Class centres sit at 0°, 120°, 240° on a circle of radius 0.4 centred at (0.5, 0.5).
	for i in 0 .. n_total {
		class_id := i % n_classes
		centre_angle := f64(class_id) * 2.0944 // 2π/3 per class
		// Deterministic noise: different for every (row, feature) pair
		noise_x := math.sin(f64(i * 7)) * 0.1
		noise_y := math.sin(f64(i * 11 + 3)) * 0.1
		x_data[i * n_features + 0] = math.cos(centre_angle) * 0.4 + 0.5 + noise_x
		x_data[i * n_features + 1] = math.sin(centre_angle) * 0.4 + 0.5 + noise_y
		y_data[i * n_classes + class_id] = 1.0
	}

	x_tensor := vtl.from_array(x_data, [n_total, n_features])!
	y_tensor := vtl.from_array(y_data, [n_total, n_classes])!

	// x does not need a gradient — only the model weights do
	mut x_all := ctx.variable(x_tensor, requires_grad: false)

	// ── Model ──────────────────────────────────────────────────────────────────
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([n_features])
	model.linear(8)
	model.relu()
	model.linear(n_classes)
	model.softmax_cross_entropy_loss()

	mut optimizer := optimizers.sgd[f64](learning_rate: learning_rate)
	optimizer.build_params(model.info.layers)

	n_batches := n_total / batch_size

	println('Training MLP on synthetic 3-class dataset (2 features, ${n_total} samples)...')
	println('Epoch | Avg Loss')
	println('------|----------')

	// ── Training loop (mini-batch SGD, following Arraymancer style) ────────────
	for epoch in 0 .. epochs {
		mut epoch_loss := f64(0)

		for b in 0 .. n_batches {
			offset := b * batch_size
			mut x_batch := x_all.slice([offset, offset + batch_size])!
			y_batch := y_tensor.slice([offset, offset + batch_size])!

			y_pred := model.forward(x_batch)!
			mut loss := model.loss(y_pred, y_batch)!

			epoch_loss += loss.value.get([0])

			loss.backprop()!
			optimizer.update()!
		}

		avg_loss := epoch_loss / f64(n_batches)
		if epoch % 100 == 0 || epoch == epochs - 1 {
			println('  ${epoch:3d} | ${avg_loss:.6f}')
		}
	}

	// ── Evaluation ─────────────────────────────────────────────────────────────
	println('')
	println('Evaluating training accuracy...')

	mut correct := 0
	for b in 0 .. n_batches {
		offset := b * batch_size
		mut x_batch := x_all.slice([offset, offset + batch_size])!
		pred := model.forward(x_batch)!

		for i in 0 .. batch_size {
			mut max_val := pred.value.get([i, 0])
			mut pred_class := 0
			for c in 1 .. n_classes {
				v := pred.value.get([i, c])
				if v > max_val {
					max_val = v
					pred_class = c
				}
			}
			row := offset + i
			mut true_class := 0
			for c in 0 .. n_classes {
				if y_data[row * n_classes + c] > 0.5 {
					true_class = c
				}
			}
			if pred_class == true_class {
				correct++
			}
		}
	}

	accuracy := f64(correct) * 100.0 / f64(n_total)
	println('Training accuracy: ${correct}/${n_total} = ${accuracy:.1f}%')
}
