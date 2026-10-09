module internal

import math
import vtl

// lstm_forward_single_with_cell runs a single unidirectional LSTM layer.
// Input uses [sequence, batch, input_size], states use [batch, hidden_size],
// and weights use PyTorch gate order [input, forget, cell, output].
pub fn lstm_forward_single_with_cell[T](input &vtl.Tensor[T], hidden0 &vtl.Tensor[T],
	cell0 &vtl.Tensor[T], w_ih &vtl.Tensor[T], w_hh &vtl.Tensor[T], b_ih &vtl.Tensor[T],
	b_hh &vtl.Tensor[T]) !(&vtl.Tensor[T], &vtl.Tensor[T], &vtl.Tensor[T]) {
	if input.shape.len != 3 || hidden0.shape.len != 2 || cell0.shape != hidden0.shape {
		return error('lstm_forward_single: expected input [sequence, batch, features] and matching [batch, hidden] states')
	}
	seq_len, batch, input_size := input.shape[0], input.shape[1], input.shape[2]
	hidden := hidden0.shape[1]
	if hidden0.shape[0] != batch || w_ih.shape != [4 * hidden, input_size]
		|| w_hh.shape != [4 * hidden, hidden] || b_ih.size() != 4 * hidden
		|| b_hh.size() != 4 * hidden {
		return error('lstm_forward_single: incompatible input, state, weight, or bias dimensions')
	}
	mut h := []f64{len: batch * hidden}
	mut c := []f64{len: batch * hidden}
	for i in 0 .. h.len {
		h[i] = f64(hidden0.get_nth(i))
		c[i] = f64(cell0.get_nth(i))
	}
	mut output := []f64{len: seq_len * batch * hidden}
	for t in 0 .. seq_len {
		for b in 0 .. batch {
			for j in 0 .. hidden {
				mut gate_values := [f64(0), f64(0), f64(0), f64(0)]
				for g in 0 .. 4 {
					row := g * hidden + j
					value := f64(b_ih.get_nth(row)) + f64(b_hh.get_nth(row))
					mut sum := value
					for k in 0 .. input_size {
						sum += f64(input.get([t, b, k])) * f64(w_ih.get([row, k]))
					}
					for k in 0 .. hidden {
						sum += h[b * hidden + k] * f64(w_hh.get([row, k]))
					}
					gate_values[g] = sum
				}
				i := lstm_sigmoid(gate_values[0])
				f := lstm_sigmoid(gate_values[1])
				g := math.tanh(gate_values[2])
				o := lstm_sigmoid(gate_values[3])
				idx := b * hidden + j
				c[idx] = f * c[idx] + i * g
				h[idx] = o * math.tanh(c[idx])
				output[t * batch * hidden + idx] = h[idx]
			}
		}
	}
	return vtl.from_array(output.map(vtl.cast[T](it)), [seq_len, batch, hidden])!, vtl.from_array(h.map(vtl.cast[T](it)), [
		batch,
		hidden,
	])!, vtl.from_array(c.map(vtl.cast[T](it)), [batch, hidden])!
}

// lstm_forward_single preserves the original zero-cell-state convenience API.
pub fn lstm_forward_single[T](input &vtl.Tensor[T], hidden0 &vtl.Tensor[T], w_ih &vtl.Tensor[T],
	w_hh &vtl.Tensor[T], b_ih &vtl.Tensor[T], b_hh &vtl.Tensor[T]) !(&vtl.Tensor[T], &vtl.Tensor[T]) {
	cell0 := vtl.zeros[T](hidden0.shape)
	output, hidden, _ := lstm_forward_single_with_cell[T](input, hidden0, cell0, w_ih, w_hh,
		b_ih, b_hh)!
	return output, hidden
}

// lstm_backward_single computes BPTT gradients for the full output sequence.
// Results are d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_hidden0, d_cell0.
pub fn lstm_backward_single[T](input &vtl.Tensor[T], hidden0 &vtl.Tensor[T],
	cell0 &vtl.Tensor[T], w_ih &vtl.Tensor[T], w_hh &vtl.Tensor[T], b_ih &vtl.Tensor[T],
	b_hh &vtl.Tensor[T], grad_output &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	output, _, _ := lstm_forward_single_with_cell[T](input, hidden0, cell0, w_ih, w_hh, b_ih,
		b_hh)!
	if grad_output.shape != output.shape {
		return error('lstm_backward_single: grad_output shape must match output')
	}
	seq_len, batch, input_size := input.shape[0], input.shape[1], input.shape[2]
	hidden := hidden0.shape[1]
	mut hs := [][]f64{len: seq_len + 1}
	mut cs := [][]f64{len: seq_len + 1}
	mut gate_is := [][]f64{len: seq_len}
	mut gate_fs := [][]f64{len: seq_len}
	mut gate_gs := [][]f64{len: seq_len}
	mut gate_os := [][]f64{len: seq_len}
	hs[0] = []f64{len: batch * hidden}
	cs[0] = []f64{len: batch * hidden}
	for i in 0 .. batch * hidden {
		hs[0][i] = f64(hidden0.get_nth(i))
		cs[0][i] = f64(cell0.get_nth(i))
	}
	for t in 0 .. seq_len {
		hs[t + 1] = []f64{len: batch * hidden}
		cs[t + 1] = []f64{len: batch * hidden}
		gate_is[t] = []f64{len: batch * hidden}
		gate_fs[t] = []f64{len: batch * hidden}
		gate_gs[t] = []f64{len: batch * hidden}
		gate_os[t] = []f64{len: batch * hidden}
		for b in 0 .. batch {
			for j in 0 .. hidden {
				mut gates := [f64(0), f64(0), f64(0), f64(0)]
				for gate in 0 .. 4 {
					row := gate * hidden + j
					sum := f64(b_ih.get_nth(row)) + f64(b_hh.get_nth(row))
					for k in 0 .. input_size {
						sum += f64(input.get([t, b, k])) * f64(w_ih.get([row, k]))
					}
					for k in 0 .. hidden {
						sum += hs[t][b * hidden + k] * f64(w_hh.get([row, k]))
					}
					gates[gate] = sum
				}
				idx := b * hidden + j
				gate_is[t][idx] = lstm_sigmoid(gates[0])
				gate_fs[t][idx] = lstm_sigmoid(gates[1])
				gate_gs[t][idx] = math.tanh(gates[2])
				gate_os[t][idx] = lstm_sigmoid(gates[3])
				cs[t + 1][idx] = gate_fs[t][idx] * cs[t][idx] + gate_is[t][idx] * gate_gs[t][idx]
				hs[t + 1][idx] = gate_os[t][idx] * math.tanh(cs[t + 1][idx])
			}
		}
	}
	mut dx := []f64{len: seq_len * batch * input_size}
	mut d_w_ih := []f64{len: 4 * hidden * input_size}
	mut d_w_hh := []f64{len: 4 * hidden * hidden}
	mut d_b_ih := []f64{len: 4 * hidden}
	mut d_b_hh := []f64{len: 4 * hidden}
	mut d_h_next := []f64{len: batch * hidden}
	mut d_c_next := []f64{len: batch * hidden}
	for rev in 0 .. seq_len {
		t := seq_len - 1 - rev
		mut d_h_prev := []f64{len: batch * hidden}
		mut d_c_prev := []f64{len: batch * hidden}
		for b in 0 .. batch {
			for j in 0 .. hidden {
				idx := b * hidden + j
				d_h := f64(grad_output.get([t, b, j])) + d_h_next[idx]
				tanh_c := math.tanh(cs[t + 1][idx])
				d_c := d_c_next[idx] + d_h * gate_os[t][idx] * (1 - tanh_c * tanh_c)
				i, f, g, o := gate_is[t][idx], gate_fs[t][idx], gate_gs[t][idx], gate_os[t][idx]
				d_gates := [d_c * g * i * (1 - i), d_c * cs[t][idx] * f * (1 - f),
					d_c * i * (1 - g * g), d_h * tanh_c * o * (1 - o)]
				d_c_prev[idx] = d_c * f
				for gate in 0 .. 4 {
					row := gate * hidden + j
					d_gate := d_gates[gate]
					d_b_ih[row] += d_gate
					d_b_hh[row] += d_gate
					for k in 0 .. input_size {
						d_w_ih[row * input_size + k] += d_gate * f64(input.get([t, b, k]))
						dx[t * batch * input_size + b * input_size + k] += d_gate * f64(w_ih.get([
							row,
							k,
						]))
					}
					for k in 0 .. hidden {
						d_w_hh[row * hidden + k] += d_gate * hs[t][b * hidden + k]
						d_h_prev[b * hidden + k] += d_gate * f64(w_hh.get([row, k]))
					}
				}
			}
		}
		d_h_next = d_h_prev
		d_c_next = d_c_prev
	}
	return [vtl.from_array(dx.map(vtl.cast[T](it)), input.shape)!,
		vtl.from_array(d_w_ih.map(vtl.cast[T](it)), w_ih.shape)!,
		vtl.from_array(d_w_hh.map(vtl.cast[T](it)), w_hh.shape)!,
		vtl.from_array(d_b_ih.map(vtl.cast[T](it)), b_ih.shape)!,
		vtl.from_array(d_b_hh.map(vtl.cast[T](it)), b_hh.shape)!,
		vtl.from_array(d_h_next.map(vtl.cast[T](it)), hidden0.shape)!,
		vtl.from_array(d_c_next.map(vtl.cast[T](it)), cell0.shape)!]
}

// lstm_forward_multi stacks independent single-layer LSTMs with zero initial states.
pub fn lstm_forward_multi[T](input &vtl.Tensor[T], h0 &vtl.Tensor[T], w_ih &vtl.Tensor[T],
	w_hh &vtl.Tensor[T], b_ih &vtl.Tensor[T], b_hh &vtl.Tensor[T]) !(&vtl.Tensor[T], &vtl.Tensor[T]) {
	if h0.shape.len != 3 || w_ih.shape.len != 3 || w_hh.shape.len != 3 || b_ih.shape.len != 2
		|| b_hh.shape.len != 2 {
		return error('lstm_forward_multi: expected stacked layer parameters')
	}
	num_layers := h0.shape[0]
	batch := input.shape[1]
	hidden := h0.shape[2]
	mut layer_input := input
	mut states := []&vtl.Tensor[T]{len: num_layers}
	for layer in 0 .. num_layers {
		lw_ih := tensor_layer_matrix[T](w_ih, layer)!
		lw_hh := tensor_layer_matrix[T](w_hh, layer)!
		lb_ih := tensor_layer_vector[T](b_ih, layer)!
		lb_hh := tensor_layer_vector[T](b_hh, layer)!
		layer_h0 := vtl.zeros[T]([batch, hidden])
		layer_c0 := vtl.zeros[T]([batch, hidden])
		layer_output, layer_h_n, _ := lstm_forward_single_with_cell[T](layer_input, layer_h0,
			layer_c0, lw_ih, lw_hh, lb_ih, lb_hh)!
		layer_input = layer_output
		states[layer] = layer_h_n
	}
	mut h_values := []f64{len: num_layers * batch * hidden}
	for layer in 0 .. num_layers {
		for i in 0 .. batch * hidden {
			h_values[layer * batch * hidden + i] = f64(states[layer].get_nth(i))
		}
	}
	return layer_input, vtl.from_array(h_values.map(vtl.cast[T](it)), [num_layers, batch, hidden])!
}

fn tensor_layer_matrix[T](tensor &vtl.Tensor[T], layer int) !&vtl.Tensor[T] {
	rows, columns := tensor.shape[1], tensor.shape[2]
	mut values := []f64{len: rows * columns}
	for r in 0 .. rows {
		for c in 0 .. columns {
			values[r * columns + c] = f64(tensor.get([layer, r, c]))
		}
	}
	return vtl.from_array(values.map(vtl.cast[T](it)), [rows, columns])
}

fn tensor_layer_vector[T](tensor &vtl.Tensor[T], layer int) !&vtl.Tensor[T] {
	width := tensor.shape[1]
	mut values := []f64{len: width}
	for i in 0 .. width {
		values[i] = f64(tensor.get([layer, i]))
	}
	return vtl.from_array(values.map(vtl.cast[T](it)), [width])
}

@[inline]
fn lstm_sigmoid(x f64) f64 {
	return 1.0 / (1.0 + math.exp(-x))
}
