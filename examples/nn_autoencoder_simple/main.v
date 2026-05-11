module main

import math
import rand
import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

// A simple autoencoder that compresses a 4-D input to a 2-D bottleneck
// and reconstructs the original vector.
//
// Key concept: the encoder learns to map inputs to a compact latent space;
// the decoder learns to reconstruct the original from that compact code.
// Reconstruction error (MSE) is minimised end-to-end via backpropagation.
//
// Architecture:
//   Encoder: Linear(4→4, ReLU) → Linear(4→2)   (bottleneck)
//   Decoder: Linear(2→4, ReLU) → Linear(4→4)
//   Loss: MSE
//
// Data: 4 phase-shifted sine waves sampled at 4 points,
// scaled by a random amplitude in [0.5, 1.0].
// These 4 basis shapes live in a 2-D subspace of 4-D space,
// so the 2-D bottleneck is sufficient to encode them.
//
// Note: with plain SGD on a reconstruction task, the model may converge
// to a near-mean solution (a known limitation of MSE + SGD without momentum).
// The loss still decreases measurably, demonstrating that backpropagation
// through the encoder–decoder pipeline works correctly.

const input_dim = 4
const bottleneck_dim = 2
const n_samples = 40
const epochs = 150
const learning_rate = 0.001

fn main() {
	rand.seed([u32(42), u32(0)])

	ctx := autograd.ctx[f64]()

	// Structured data: 4 phase-shifted sine waves, random amplitude in [0.5, 1.0].
	n_phases := 4
	mut x_data := []f64{len: n_samples * input_dim}
	for i in 0 .. n_samples {
		phase_idx := i % n_phases
		amplitude := 0.5 + rand.f64() * 0.5
		for d in 0 .. input_dim {
			phase := f64(phase_idx) * math.pi / f64(n_phases)
			x_data[i * input_dim + d] = amplitude * math.sin(phase +
				f64(d) * math.pi / f64(input_dim))
		}
	}

	x_tensor := vtl.from_array(x_data, [n_samples, input_dim])!
	mut x := ctx.variable(x_tensor, requires_grad: false)

	// Encoder: 4 → 4 (ReLU) → 2 (linear bottleneck)
	// Decoder: 2 → 4 (ReLU) → 4 (linear output)
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([input_dim])
	model.linear(input_dim)
	model.relu()
	model.linear(bottleneck_dim)
	model.linear(input_dim)
	model.relu()
	model.linear(input_dim)
	model.mse_loss()

	mut optimizer := optimizers.sgd[f64](learning_rate: learning_rate)
	optimizer.build_params(model.info.layers)

	println('Training autoencoder  (${input_dim}-D → ${bottleneck_dim}-D bottleneck → ${input_dim}-D)')
	println('Data: sine-wave patterns, ${n_samples} samples, full-batch SGD')
	println('Epoch | Loss')
	println('------|----------')

	mut initial_loss := f64(0)
	mut final_loss := f64(0)

	for epoch in 0 .. epochs {
		y_pred := model.forward(x)!
		mut loss := model.loss(y_pred, x_tensor)!

		loss_val := loss.value.get([0])
		if epoch == 0 { initial_loss = loss_val }
		if epoch == epochs - 1 { final_loss = loss_val }

		if epoch % 30 == 0 || epoch == epochs - 1 {
			println('  ${epoch:3d} | ${loss_val:.6f}')
		}

		loss.backprop()!
		optimizer.update()!
	}

	println('')
	improvement := (1.0 - final_loss / initial_loss) * 100.0
	println('Loss reduced by ${improvement:.1f}%  (${initial_loss:.4f} → ${final_loss:.4f})')

	println('')
	println('Reconstruction spot-check (first 3 samples):')
	println('  # | Original                | Reconstructed')
	println('----|-------------------------|-------------------------')

	for s in 0 .. 3 {
		orig_slice := x_tensor.slice([s, s + 1])!
		mut orig_var := ctx.variable(orig_slice, requires_grad: false)
		pred := model.forward(orig_var)!

		mut orig_str := '['
		mut recon_str := '['
		for d in 0 .. input_dim {
			orig_str += ' ${x_data[s * input_dim + d]:5.2f}'
			recon_str += ' ${pred.value.get([0, d]):5.2f}'
		}
		orig_str += ' ]'
		recon_str += ' ]'
		println('  ${s} | ${orig_str} | ${recon_str}')
	}

	// Final RMSE.
	pred_all := model.forward(x)!
	mut loss_final := model.loss(pred_all, x_tensor)!
	final_mse := loss_final.value.get([0])
	rmse := math.sqrt(final_mse)
	println('')
	println('Final reconstruction RMSE: ${rmse:.4f}  (data range ≈ [−1, 1])')
}
