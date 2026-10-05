module internal

import math
import vtl

// gru_forward_single computes a single-layer, unidirectional GRU.
// Input is [sequence, batch, input_size]. Weights follow PyTorch's gate
// order [reset, update, new] and shapes [3*hidden_size, input_size/hidden_size].
pub fn gru_forward_single[T](input &vtl.Tensor[T], w_ih &vtl.Tensor[T], w_hh &vtl.Tensor[T],
	b_ih &vtl.Tensor[T], b_hh &vtl.Tensor[T], h0 &vtl.Tensor[T]) !(&vtl.Tensor[T], &vtl.Tensor[T]) {
	if input.shape.len != 3 || h0.shape.len != 2 {
		return error('gru_forward_single: expected input [sequence, batch, features] and h0 [batch, hidden]')
	}
	seq_len, batch, input_size := input.shape[0], input.shape[1], input.shape[2]
	hidden := h0.shape[1]
	if w_ih.shape != [3 * hidden, input_size] || w_hh.shape != [3 * hidden, hidden]
		|| b_ih.size() != 3 * hidden || b_hh.size() != 3 * hidden
		|| h0.shape[0] != batch {
		return error('gru_forward_single: incompatible input, state, weight, or bias dimensions')
	}
	mut state := []f64{len: batch * hidden}
	for i in 0 .. state.len { state[i] = f64(h0.get_nth(i)) }
	mut output := []f64{len: seq_len * batch * hidden}
	for t in 0 .. seq_len {
		mut next := []f64{len: batch * hidden}
		for b in 0 .. batch {
			for j in 0 .. hidden {
				mut r_x, mut z_x, mut n_x := f64(0), f64(0), f64(0)
				mut r_h, mut z_h, mut n_h := f64(0), f64(0), f64(0)
				for k in 0 .. input_size {
					x := f64(input.get([t, b, k]))
					r_x += x * f64(w_ih.get([j, k]))
					z_x += x * f64(w_ih.get([hidden + j, k]))
					n_x += x * f64(w_ih.get([2 * hidden + j, k]))
				}
				for k in 0 .. hidden {
					h := state[b * hidden + k]
					r_h += h * f64(w_hh.get([j, k]))
					z_h += h * f64(w_hh.get([hidden + j, k]))
					n_h += h * f64(w_hh.get([2 * hidden + j, k]))
				}
				r := sigmoid_value(r_x + f64(b_ih.get_nth(j)) + r_h + f64(b_hh.get_nth(j)))
				z := sigmoid_value(z_x + f64(b_ih.get_nth(hidden + j)) + z_h + f64(b_hh.get_nth(hidden + j)))
				n := math.tanh(n_x + f64(b_ih.get_nth(2 * hidden + j)) + r * (n_h + f64(b_hh.get_nth(2 * hidden + j))))
				value := (1 - z) * n + z * state[b * hidden + j]
				next[b * hidden + j] = value
				output[t * batch * hidden + b * hidden + j] = value
			}
		}
		state = next
	}
	return vtl.from_array(output.map(vtl.cast[T](it)), [seq_len, batch, hidden])!, vtl.from_array(state.map(vtl.cast[T](it)), [
		batch,
		hidden,
	])!
}

// gru_backward_single differentiates the sum over timesteps supplied in grad_output.
// It returns gradients for input, w_ih, w_hh, b_ih, and b_hh respectively.
pub fn gru_backward_single[T](input &vtl.Tensor[T], w_ih &vtl.Tensor[T], w_hh &vtl.Tensor[T],
	b_ih &vtl.Tensor[T], b_hh &vtl.Tensor[T], h0 &vtl.Tensor[T], grad_output &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	output, _ := gru_forward_single[T](input, w_ih, w_hh, b_ih, b_hh, h0)!
	if grad_output.shape != output.shape {
		return error('gru_backward_single: grad_output shape must match output')
	}
	seq_len, batch, input_size := input.shape[0], input.shape[1], input.shape[2]
	hidden := h0.shape[1]
	mut states := [][]f64{len: seq_len + 1}
	states[0] = []f64{len: batch * hidden}
	for i in 0 .. batch * hidden { states[0][i] = f64(h0.get_nth(i)) }
	mut reset := [][]f64{len: seq_len}
	mut update := [][]f64{len: seq_len}
	mut candidate := [][]f64{len: seq_len}
	mut hidden_candidate := [][]f64{len: seq_len}
	for t in 0 .. seq_len {
		states[t + 1] = []f64{len: batch * hidden}
		reset[t] = []f64{len: batch * hidden}
		update[t] = []f64{len: batch * hidden}
		candidate[t] = []f64{len: batch * hidden}
		hidden_candidate[t] = []f64{len: batch * hidden}
		for b in 0 .. batch {
			for j in 0 .. hidden {
				mut rx, mut zx, mut nx, mut rh, mut zh, mut nh := f64(0), f64(0), f64(0), f64(0), f64(0), f64(0)
				for k in 0 .. input_size {
					x := f64(input.get([t, b, k]))
					rx += x * f64(w_ih.get([j, k]))
					zx += x * f64(w_ih.get([hidden + j, k]))
					nx += x * f64(w_ih.get([2 * hidden + j, k]))
				}
				for k in 0 .. hidden {
					h := states[t][b * hidden + k]
					rh += h * f64(w_hh.get([j, k]))
					zh += h * f64(w_hh.get([hidden + j, k]))
					nh += h * f64(w_hh.get([2 * hidden + j, k]))
				}
				idx := b * hidden + j
				r := sigmoid_value(rx + f64(b_ih.get_nth(j)) + rh + f64(b_hh.get_nth(j)))
				z := sigmoid_value(zx + f64(b_ih.get_nth(hidden + j)) + zh + f64(b_hh.get_nth(hidden + j)))
				q := nh + f64(b_hh.get_nth(2 * hidden + j))
				n := math.tanh(nx + f64(b_ih.get_nth(2 * hidden + j)) + r * q)
				reset[t][idx], update[t][idx], candidate[t][idx], hidden_candidate[t][idx] = r, z, n, q
				states[t + 1][idx] = (1 - z) * n + z * states[t][idx]
			}
		}
	}
	mut dx := []f64{len: seq_len * batch * input_size}
	mut dwx := []f64{len: 3 * hidden * input_size}
	mut dwh := []f64{len: 3 * hidden * hidden}
	mut dbx := []f64{len: 3 * hidden}
	mut dbh := []f64{len: 3 * hidden}
	mut dh_next := []f64{len: batch * hidden}
	for rev in 0 .. seq_len {
		t := seq_len - 1 - rev
		dh_future := dh_next.clone()
		dh_next = []f64{len: batch * hidden}
		for b in 0 .. batch {
			for j in 0 .. hidden {
				idx := b * hidden + j
				r, z, n, q := reset[t][idx], update[t][idx], candidate[t][idx], hidden_candidate[t][idx]
				hprev := states[t][idx]
				dh := f64(grad_output.get([t, b, j])) + dh_future[idx]
				dnpre := dh * (1 - z) * (1 - n * n)
				dzpre := dh * (hprev - n) * z * (1 - z)
				drpre := dnpre * q * r * (1 - r)
				dinput := [drpre, dzpre, dnpre]
				dhidden := [drpre, dzpre, dnpre * r]
				dh_next[idx] += dh * z
				for g in 0 .. 3 {
					row := g * hidden + j
					dx_gate := dinput[g]
					dh_gate := dhidden[g]
					dbx[row] += dx_gate
					dbh[row] += dh_gate
					for k in 0 .. input_size {
						dwx[row * input_size + k] += dx_gate * f64(input.get([t, b, k]))
						dx[t * batch * input_size + b * input_size + k] += dx_gate * f64(w_ih.get([
							row,
							k,
						]))
					}
					for k in 0 .. hidden {
						dwh[row * hidden + k] += dh_gate * states[t][b * hidden + k]
						dh_next[b * hidden + k] += dh_gate * f64(w_hh.get([row, k]))
					}
				}
			}
		}
	}
	return [vtl.from_array(dx.map(vtl.cast[T](it)), [seq_len, batch, input_size])!,
		vtl.from_array(dwx.map(vtl.cast[T](it)), [3 * hidden, input_size])!,
		vtl.from_array(dwh.map(vtl.cast[T](it)), [3 * hidden, hidden])!,
		vtl.from_array(dbx.map(vtl.cast[T](it)), [3 * hidden])!,
		vtl.from_array(dbh.map(vtl.cast[T](it)), [3 * hidden])!]
}

@[inline]
fn sigmoid_value(x f64) f64 { return 1.0 / (1.0 + math.exp(-x)) }
