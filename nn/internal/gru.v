module internal

import vtl
import vtl.la

// gru_forward_single runs a single-layer GRU over a sequence.
//
// Implements the standard GRU equations (PyTorch/CuDNN compatible):
//   r_t = sigmoid(x_t @ W_ir^T + h_{t-1} @ W_hr^T + b_ir + b_hr)
//   z_t = sigmoid(x_t @ W_iz^T + h_{t-1} @ W_hz^T + b_iz + b_hz)
//   n_t = tanh(x_t @ W_in^T + b_in + r_t * (h_{t-1} @ W_hn^T + b_hn))
//   h_t = (1 - z_t) * n_t + z_t * h_{t-1}
//
// Shapes:
//   input:   [seq_len, batch, input_size]
//   hidden0: [batch, hidden_size]
//   w_ih:    [3*hidden_size, input_size]  (r, z, n gates stacked)
//   w_hh:    [3*hidden_size, hidden_size]
//   b_ih:    [3*hidden_size]
//   b_hh:    [3*hidden_size]
//
// Returns (output [seq_len, batch, hidden_size], h_n [batch, hidden_size])
pub fn gru_forward_single[T](input &vtl.Tensor[T],
	hidden0 &vtl.Tensor[T],
	w_ih &vtl.Tensor[T],
	w_hh &vtl.Tensor[T],
	b_ih &vtl.Tensor[T],
	b_hh &vtl.Tensor[T]) !(&vtl.Tensor[T], &vtl.Tensor[T]) {
	seq_len := input.shape[0]
	batch := input.shape[1]
	hidden_size := hidden0.shape[1]

	w_ih_t := w_ih.transpose([1, 0])!
	w_hh_t := w_hh.transpose([1, 0])!

	mut h := hidden0
	mut all_outputs := []f64{len: seq_len * batch * hidden_size}

	for t in 0 .. seq_len {
		// Extract x_t: [batch, input_size]
		x_t_size := input.shape[2]
		mut x_t_data := []f64{len: batch * x_t_size}
		for b in 0 .. batch {
			for f in 0 .. x_t_size {
				x_t_data[b * x_t_size + f] = f64(input.get([t, b, f]))
			}
		}
		x_t := vtl.from_array(x_t_data.map(vtl.cast[T](it)), [batch, x_t_size])!

		// Compute input gates: x_t @ W_ih^T → [batch, 3*hidden_size]
		gate_ih := la.matmul[T](x_t, w_ih_t)!
		// Compute hidden gates: h @ W_hh^T → [batch, 3*hidden_size]
		gate_hh := la.matmul[T](h, w_hh_t)!

		mut h_new_data := []f64{len: batch * hidden_size}

		for b in 0 .. batch {
			for idx in 0 .. hidden_size {
				// Input-side gate values with bias
				mut r_ih := f64(gate_ih.get([b, idx]))
				mut z_ih := f64(gate_ih.get([b, hidden_size + idx]))
				mut n_ih := f64(gate_ih.get([b, 2 * hidden_size + idx]))

				// Hidden-side gate values with bias
				mut r_hh := f64(gate_hh.get([b, idx]))
				mut z_hh := f64(gate_hh.get([b, hidden_size + idx]))
				mut n_hh := f64(gate_hh.get([b, 2 * hidden_size + idx]))

				// Add biases
				if b_ih != unsafe { nil } {
					r_ih += f64(b_ih.get_nth(idx))
					z_ih += f64(b_ih.get_nth(hidden_size + idx))
					n_ih += f64(b_ih.get_nth(2 * hidden_size + idx))
				}
				if b_hh != unsafe { nil } {
					r_hh += f64(b_hh.get_nth(idx))
					z_hh += f64(b_hh.get_nth(hidden_size + idx))
					n_hh += f64(b_hh.get_nth(2 * hidden_size + idx))
				}

				// Gate activations
				r_gate := gru_sigmoid(r_ih + r_hh)
				z_gate := gru_sigmoid(z_ih + z_hh)
				n_gate := gru_tanh(n_ih + r_gate * n_hh)

				// GRU update: h_t = (1 - z) * n + z * h_{t-1}
				h_prev := f64(h.get([b, idx]))
				h_new_data[b * hidden_size + idx] = (1.0 - z_gate) * n_gate + z_gate * h_prev
			}
		}

		h = vtl.from_array(h_new_data.map(vtl.cast[T](it)), [batch, hidden_size])!

		// Store h_t in output
		for b in 0 .. batch {
			for idx in 0 .. hidden_size {
				all_outputs[t * batch * hidden_size + b * hidden_size + idx] = h_new_data[
					b * hidden_size + idx]
			}
		}
	}

	output := vtl.from_array(all_outputs.map(vtl.cast[T](it)), [seq_len, batch, hidden_size])!
	return output, h
}

// gru_sigmoid computes logistic sigmoid for GRU gate activations.
@[inline]
fn gru_sigmoid(x f64) f64 {
	return 1.0 / (1.0 + gru_exp(-x))
}

// gru_tanh computes hyperbolic tangent for GRU gate activations.
@[inline]
fn gru_tanh(x f64) f64 {
	if x > 20.0 {
		return 1.0
	}
	if x < -20.0 {
		return -1.0
	}
	e2x := gru_exp(2.0 * x)
	return (e2x - 1.0) / (e2x + 1.0)
}

// gru_exp is a fast f64 exp approximation (avoids importing math).
@[inline]
fn gru_exp(x f64) f64 {
	mut result := f64(1.0)
	mut term := f64(1.0)
	for n in 1 .. 20 {
		term *= x / f64(n)
		result += term
	}
	return result
}
