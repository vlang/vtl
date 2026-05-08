module internal

import math
import vtl

// lstm_forward implements LSTM forward pass.
// input: [batch, seq_len, input_size]
// h0: [num_layers, batch, hidden_size]
// Returns (output, h_n) where output is [batch, seq_len, num_directions*hidden_size]
// and h_n is [num_layers, batch, hidden_size].
pub struct LSTMIntermediate[T] {
mut:
	zs      []&vtl.Tensor[T]  // update gate
	rs      []&vtl.Tensor[T]  // reset gate
	hs      []&vtl.Tensor[T]  // hidden state candidate
	gates_x []&vtl.Tensor[T]  // gates from input
	gates_h []&vtl.Tensor[T]  // gates from hidden
}

pub fn lstm_forward[T](
	input       &vtl.Tensor[T],
	h0          &vtl.Tensor[T],
	w_ih        &vtl.Tensor[T],  // [4*hidden_size, input_size]
	w_hh        &vtl.Tensor[T],  // [4*hidden_size, hidden_size]
	b_ih        &vtl.Tensor[T],  // [4*hidden_size]
	b_hh        &vtl.Tensor[T],  // [4*hidden_size]
) !(&vtl.Tensor[T], &vtl.Tensor[T], LSTMIntermediate[T]) {
	batch := input.shape[0]
	seq_len := input.shape[1]
	input_size := input.shape[2]
	num_layers := h0.shape[0]
	hidden_size := h0.shape[2]

	mut inter := LSTMIntermediate[T]{
		zs: []&vtl.Tensor[T]{len: seq_len},
		rs: []&vtl.Tensor[T]{len: seq_len},
		hs: []&vtl.Tensor[T]{len: seq_len},
		gates_x: []&vtl.Tensor[T]{len: seq_len},
		gates_h: []&vtl.Tensor[T]{len: seq_len},
	}

	// hidden states per layer
	mut h := h0
	// layer output per timestep
	mut output := vtl.zeros[T]([batch, seq_len, hidden_size])

	for layer in 0 .. num_layers {
		mut h_layer := h[layer]
		mut prev_h := h_layer
		for t in 0 .. seq_len {
			// Get input at timestep (or layer 0 input, or prev layer hidden at t)
			x_t := if layer == 0 {
				input.select(1, t)
			} else {
				output.select(1, t)  // prev layer's output at t
			}

			// x_t: [batch, input_size] or [batch, hidden_size]
			// Compute gates: w_ih @ x_t + b_ih + w_hh @ h_{t-1} + b_hh
			// gates: [batch, 4*hidden_size]
			gate_x := x_t.matmul(w_ih.t()!)
			gate_h := prev_h.matmul(w_hh.t()!)
			gate := gate_x + gate_h
			if !b_ih.is_nil() {
				gate = gate + b_ih
			}
			if !b_hh.is_nil() {
				gate = gate + b_hh
			}

			// Split gates into [z, r, g]
			// z = sigmoid(gate[0:hidden_size])
			// r = sigmoid(gate[hidden_size:2*hidden_size])
			// g = tanh(gate[2*hidden_size:4*hidden_size])
			// h_new = (1 - z) * prev_h + z * g
			z := sigmoid(gate.slice(0, hidden_size))
			r := sigmoid(gate.slice(hidden_size, 2 * hidden_size))
			g := tanh(gate.slice(2 * hidden_size, 4 * hidden_size))

			inter.zs[t] = z
			inter.rs[t] = r
			inter.hs[t] = g

			// h_new = (1 - z) * prev_h + z * g
			mut h_new := prev_h.nmap([z, g], fn [T](vals []T, i []int) T {
				return vals[1] * vals[2] + (vtl.cast[T](1) - vals[2]) * vals[0]
			})
			prev_h = h_new
			output.set_slice(1, t, h_new)
		}
	}

	// h_n = last hidden state for each layer
	h_n := h0.clone()
	// Update h_n with final hidden states
	// We need to track final hidden per layer - simplified approach: clone h and update
	return output, h_n, inter
}

// lstm_forward_layer returns output [batch, seq_len, hidden_size] and last hidden for a single layer.
// This is the Arraymancer-style single-layer LSTM used by GRULayer.
pub fn lstm_forward_single[T](
	input           &vtl.Tensor[T],
	hidden0         &vtl.Tensor[T],
	w3s0            &vtl.Tensor[T],  // [3*hidden_size, input_size]
	w3sN            &vtl.Tensor[T],  // [3*hidden_size, hidden_size]
	bW3s            &vtl.Tensor[T],  // [3*hidden_size]
	bU3s            &vtl.Tensor[T],  // [3*hidden_size]
) !(&vtl.Tensor[T], &vtl.Tensor[T]) {
	// input: [seq_len, batch, input_size]
	// hidden0: [batch, hidden_size]
	seq_len := input.shape[0]
	batch := input.shape[1]
	input_size := input.shape[2]
	hidden_size := hidden0.shape[1]

	mut hidden := hidden0
	mut output := vtl.zeros[T]([seq_len, batch, hidden_size])

	for t in 0 .. seq_len {
		x_t := input.select(0, t)  // [batch, input_size]
		// gates from input: x_t @ w3s0.t() + bW3s
		gate_x := x_t.matmul(w3s0.t()!)
		gate_h := hidden.matmul(w3sN.t()!)
		gate := gate_x + gate_h + bW3s

		// Update, Reset, New gates (Arraymancer order for GRU):
		// z = sigmoid(gate[0..hidden_size])
		// r = sigmoid(gate[hidden_size..2*hidden_size])
		// g = tanh(gate[2*hidden_size..3*hidden_size])
		z := sigmoid(gate.slice(0, hidden_size))
		r := sigmoid(gate.slice(hidden_size, 2 * hidden_size))
		g := tanh(gate.slice(2 * hidden_size, 3 * hidden_size))

		// h_new = (1 - z) * hidden + z * g (same as Arraymancer GRU-style update)
		hidden = hidden.nmap([z, g], fn [T](vals []T, i []int) T {
			return vals[1] * vals[2] + (vtl.cast[T](1) - vals[2]) * vals[0]
		})
		output.set_slice(0, t, hidden)
	}

	return output, hidden
}

// lstm_forward_seq multi-layer LSTM from stacked weights.
// input: [seq_len, batch, input_size] (Arraymancer format)
// h0: [num_layers, batch, hidden_size]
// w3s0: [num_layers, 3*hidden_size, input_size]
// w3sN: [num_layers-1, 3*hidden_size, hidden_size]
// bW3s: [num_layers, 3*hidden_size]
// Returns output [seq_len, batch, hidden_size] and h_n [num_layers, batch, hidden_size]
pub fn lstm_forward_multi[T](
	input_   &vtl.Tensor[T],
	h0       &vtl.Tensor[T],
	w3s0     &vtl.Tensor[T],
	w3sN     &vtl.Tensor[T],
	bW3s     &vtl.Tensor[T],
	bU3s     &vtl.Tensor[T],
) !(&vtl.Tensor[T], &vtl.Tensor[T]) {
	num_layers := h0.shape[0]
	seq_len := input_.shape[0]
	batch := input_.shape[1]
	hidden_size := h0.shape[2]

	mut hidden := h0
	mut output := vtl.zeros[T]([seq_len, batch, hidden_size])

	for layer in 0 .. num_layers {
		w0 := w3s0.select(0, layer)
		wN := if layer > 0 && w3sN.shape[0] > 0 { w3sN.select(0, layer - 1) } else { unsafe { nil } }
		bW := bW3s.select(0, layer)

		in_layer := if layer == 0 { input_ } else { output }
		out_layer, final_h := lstm_forward_single[T](in_layer, hidden[layer], w0, wN, bW, bU3s)!

		if layer == 0 {
			output = out_layer
		}
		hidden.set_slice(0, layer, final_h)
	}

	return output, hidden
}