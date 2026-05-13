module internal

import vtl
import vtl.la

// LSTMIntermediate stores per-timestep gate values for backprop.
pub struct LSTMIntermediate[T] {
mut:
	gates   []&vtl.Tensor[T] // full gate tensor per timestep
	cells   []&vtl.Tensor[T] // cell state per timestep
	hiddens []&vtl.Tensor[T] // hidden state per timestep
}

// lstm_forward_single runs a single-layer LSTM over a sequence.
//
// Implements the standard LSTM equations:
//   i_t = sigmoid(x_t @ W_ii^T + h_{t-1} @ W_hi^T + b_ii + b_hi)
//   f_t = sigmoid(x_t @ W_if^T + h_{t-1} @ W_hf^T + b_if + b_hf)
//   g_t = tanh(x_t @ W_ig^T + h_{t-1} @ W_hg^T + b_ig + b_hg)
//   o_t = sigmoid(x_t @ W_io^T + h_{t-1} @ W_ho^T + b_io + b_ho)
//   c_t = f_t * c_{t-1} + i_t * g_t
//   h_t = o_t * tanh(c_t)
//
// Shapes:
//   input:   [seq_len, batch, input_size]
//   hidden0: [batch, hidden_size]
//   cell0:   [batch, hidden_size]
//   w_ih:    [4*hidden_size, input_size]
//   w_hh:    [4*hidden_size, hidden_size]
//   b_ih:    [4*hidden_size]
//   b_hh:    [4*hidden_size]
//
// Returns (output [seq_len, batch, hidden_size], h_n [batch, hidden_size], c_n [batch, hidden_size])
pub fn lstm_forward_single[T](input &vtl.Tensor[T],
	hidden0 &vtl.Tensor[T],
	cell0 &vtl.Tensor[T],
	w_ih &vtl.Tensor[T],
	w_hh &vtl.Tensor[T],
	b_ih &vtl.Tensor[T],
	b_hh &vtl.Tensor[T]) !(&vtl.Tensor[T], &vtl.Tensor[T], &vtl.Tensor[T]) {
	seq_len := input.shape[0]
	batch := input.shape[1]
	hidden_size := hidden0.shape[1]

	w_ih_t := w_ih.transpose([1, 0])!
	w_hh_t := w_hh.transpose([1, 0])!

	mut h := hidden0
	mut c := cell0
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

		// gates = x_t @ W_ih^T + h @ W_hh^T + b_ih + b_hh
		gate_ih := la.matmul[T](x_t, w_ih_t)!
		gate_hh := la.matmul[T](h, w_hh_t)!
		gate := gate_ih.add(gate_hh)!

		gate_sz := 4 * hidden_size
		mut gate_data := []f64{len: batch * gate_sz}
		for b in 0 .. batch {
			for g in 0 .. gate_sz {
				mut v := f64(gate.get([b, g]))
				if b_ih != unsafe { nil } {
					v += f64(b_ih.get_nth(g))
				}
				if b_hh != unsafe { nil } {
					v += f64(b_hh.get_nth(g))
				}
				gate_data[b * gate_sz + g] = v
			}
		}

		// Split into 4 gates: i, f, g, o (each [batch, hidden_size])
		mut h_new_data := []f64{len: batch * hidden_size}
		mut c_new_data := []f64{len: batch * hidden_size}
		for b in 0 .. batch {
			for idx in 0 .. hidden_size {
				i_val := gate_data[b * gate_sz + idx]
				f_val := gate_data[b * gate_sz + hidden_size + idx]
				g_val := gate_data[b * gate_sz + 2 * hidden_size + idx]
				o_val := gate_data[b * gate_sz + 3 * hidden_size + idx]

				i_gate := lstm_sigmoid(i_val)
				f_gate := lstm_sigmoid(f_val)
				g_gate := vtl_tanh(g_val)
				o_gate := lstm_sigmoid(o_val)

				c_prev := f64(c.get([b, idx]))
				c_new := f_gate * c_prev + i_gate * g_gate
				h_new := o_gate * vtl_tanh(c_new)

				c_new_data[b * hidden_size + idx] = c_new
				h_new_data[b * hidden_size + idx] = h_new
			}
		}

		h = vtl.from_array(h_new_data.map(vtl.cast[T](it)), [batch, hidden_size])!
		c = vtl.from_array(c_new_data.map(vtl.cast[T](it)), [batch, hidden_size])!

		// Store h_t in output
		for b in 0 .. batch {
			for idx in 0 .. hidden_size {
				all_outputs[t * batch * hidden_size + b * hidden_size + idx] = h_new_data[
					b * hidden_size + idx]
			}
		}
	}

	output := vtl.from_array(all_outputs.map(vtl.cast[T](it)), [seq_len, batch, hidden_size])!
	return output, h, c
}

// lstm_sigmoid computes the logistic sigmoid function for gate activations.
@[inline]
fn lstm_sigmoid(x f64) f64 {
	return 1.0 / (1.0 + vtl_exp(-x))
}

// vtl_exp is a helper for f64 exp (avoids importing math in closures).
@[inline]
fn vtl_exp(x f64) f64 {
	mut result := f64(1.0)
	mut term := f64(1.0)
	for n in 1 .. 20 {
		term *= x / f64(n)
		result += term
	}
	return result
}

// vtl_tanh is a helper for f64 tanh.
@[inline]
fn vtl_tanh(x f64) f64 {
	if x > 20.0 {
		return 1.0
	}
	if x < -20.0 {
		return -1.0
	}
	e2x := vtl_exp(2.0 * x)
	return (e2x - 1.0) / (e2x + 1.0)
}
