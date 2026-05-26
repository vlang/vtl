module internal

import vtl
import vtl.la

// LSTMIntermediate stores per-timestep gate values for backprop.
pub struct LSTMIntermediate[T] {
mut:
	zs      []&vtl.Tensor[T] // update gate per timestep
	rs      []&vtl.Tensor[T] // reset gate per timestep
	hs      []&vtl.Tensor[T] // candidate hidden per timestep
	gates_x []&vtl.Tensor[T] // gates from input per timestep
	gates_h []&vtl.Tensor[T] // gates from hidden per timestep
}

// lstm_forward_single runs a single-layer LSTM.
// input: [seq_len, batch, input_size]
// hidden0: [batch, hidden_size]
// w_ih: [4*hidden_size, input_size]  (i, f, g, o gates)
// w_hh: [4*hidden_size, hidden_size]
// b_ih, b_hh: [4*hidden_size]
// Returns (output [seq_len, batch, hidden_size], h_n [batch, hidden_size])
pub fn lstm_forward_single[T](input &vtl.Tensor[T],
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

	unsafe {
		mut h := hidden0
		mut all_outputs := []f64{len: seq_len * batch * hidden_size}

		for t in 0 .. seq_len {
			x_t_size := input.shape[2]
			mut x_t_data := []f64{len: batch * x_t_size}
			for b in 0 .. batch {
				for f in 0 .. x_t_size {
					x_t_data[b * x_t_size + f] = f64(input.get([t, b, f]))
				}
			}
			x_t := vtl.from_array(x_t_data.map(vtl.cast[T](it)), [batch, x_t_size])!

			gate_ih := la.matmul[T](x_t, w_ih_t)!
			gate_hh := la.matmul[T](h, w_hh_t)!
			gate := gate_ih.add(gate_hh)!

			gate_sz := 4 * hidden_size
			mut gate_data := []f64{len: batch * gate_sz}
			for b in 0 .. batch {
				for g in 0 .. gate_sz {
					mut v := f64(gate.get([b, g]))
					if !isnil(b_ih) {
						v += f64(b_ih.get_nth(g))
					}
					if !isnil(b_hh) {
						v += f64(b_hh.get_nth(g))
					}
					gate_data[b * gate_sz + g] = v
				}
			}
			gate_full := vtl.from_array(gate_data.map(vtl.cast[T](it)), [batch, gate_sz])!

			mut h_new_data := []f64{len: batch * hidden_size}
			for b in 0 .. batch {
				for idx in 0 .. hidden_size {
					i_gate := 1.0 / (1.0 + vtl_exp(-f64(gate_full.get([b, idx]))))
					f_gate := 1.0 / (1.0 + vtl_exp(-f64(gate_full.get([b, hidden_size + idx]))))
					g_gate := vtl_tanh(f64(gate_full.get([b, 2 * hidden_size + idx])))
					o_gate := 1.0 / (1.0 + vtl_exp(-f64(gate_full.get([b, 3 * hidden_size + idx]))))
					h_prev := f64(h.get([b, idx]))
					h_new_data[b * hidden_size + idx] = o_gate * vtl_tanh(f_gate * h_prev +
						i_gate * g_gate)
				}
			}
			h = vtl.from_array(h_new_data.map(vtl.cast[T](it)), [batch, hidden_size])!

			for b in 0 .. batch {
				for idx in 0 .. hidden_size {
					all_outputs[t * batch * hidden_size + b * hidden_size + idx] = f64(h.get([
						b,
						idx,
					]))
				}
			}
		}

		output := vtl.from_array(all_outputs.map(vtl.cast[T](it)), [seq_len, batch, hidden_size])!
		return output, h
	}
}

// lstm_forward_multi stacks multiple LSTM layers.
pub fn lstm_forward_multi[T](input_ &vtl.Tensor[T],
	h0 &vtl.Tensor[T],
	w_ih &vtl.Tensor[T],
	w_hh &vtl.Tensor[T],
	b_ih &vtl.Tensor[T],
	b_hh &vtl.Tensor[T]) !(&vtl.Tensor[T], &vtl.Tensor[T]) {
	num_layers := h0.shape[0]
	batch := input_.shape[1]
	hidden_size := h0.shape[2]

	mut layer_input := input_
	mut h_list := []&vtl.Tensor[T]{len: num_layers}

	for layer in 0 .. num_layers {
		layer_w_ih_sz := w_ih.size() / num_layers
		layer_w_hh_sz := w_hh.size() / num_layers
		rows_ih := w_ih.shape[1]
		cols_ih := w_ih.shape[2]
		rows_hh := w_hh.shape[1]
		cols_hh := w_hh.shape[2]

		mut lw_ih_data := []f64{len: layer_w_ih_sz}
		mut lw_hh_data := []f64{len: layer_w_hh_sz}
		for r in 0 .. rows_ih {
			for c in 0 .. cols_ih {
				lw_ih_data[r * cols_ih + c] = f64(w_ih.get([layer, r, c]))
			}
		}
		for r in 0 .. rows_hh {
			for c in 0 .. cols_hh {
				lw_hh_data[r * cols_hh + c] = f64(w_hh.get([layer, r, c]))
			}
		}
		lw_ih := vtl.from_array(lw_ih_data.map(vtl.cast[T](it)), [rows_ih, cols_ih])!
		lw_hh := vtl.from_array(lw_hh_data.map(vtl.cast[T](it)), [rows_hh, cols_hh])!

		mut lb_ih_data := []f64{len: 4 * hidden_size}
		mut lb_hh_data := []f64{len: 4 * hidden_size}
		for g in 0 .. 4 * hidden_size {
			lb_ih_data[g] = f64(b_ih.get([layer, g]))
			lb_hh_data[g] = f64(b_hh.get([layer, g]))
		}
		lb_ih := vtl.from_array(lb_ih_data.map(vtl.cast[T](it)), [4 * hidden_size])!
		lb_hh := vtl.from_array(lb_hh_data.map(vtl.cast[T](it)), [4 * hidden_size])!

		layer_h0 := vtl.from_array([]f64{len: batch * hidden_size}.map(fn (_ f64) f64 { 0 }), [batch, hidden_size])!
		layer_output, layer_h_n := lstm_forward_single[T](layer_input, layer_h0, lw_ih, lw_hh, lb_ih, lb_hh)!
		layer_input = layer_output
		h_list[layer] = layer_h_n
	}

	output := layer_input
	mut h_n_data := []f64{len: num_layers * batch * hidden_size}
	for layer in 0 .. num_layers {
		for b in 0 .. batch {
			for idx in 0 .. hidden_size {
				h_n_data[layer * batch * hidden_size + b * hidden_size + idx] = f64(h_list[layer].get([b, idx]))
			}
		}
	}
	h_n := vtl.from_array(h_n_data.map(vtl.cast[T](it)), [num_layers, batch, hidden_size])!
	return output, h_n
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
	e2x := vtl_exp(2.0 * x)
	return (e2x - f64(1.0)) / (e2x + f64(1.0))
}