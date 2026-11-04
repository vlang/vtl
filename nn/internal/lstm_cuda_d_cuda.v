module internal

import math
import vsl.cuda
import vtl
import vtl.autograd_cuda

// LSTM forward and BPTT backward on cuBLAS (ADR-002, phase 2).
//
// ADR-002 chose the cuDNN RNN API for LSTM. This version composes the recurrence from
// cuBLAS GEMMs instead, and the ADR records that change. The cuDNN RNN backward needs a
// cuDNN training forward pass to produce its reserve space, and it needs our weights
// packed into cuDNN's internal layout. Composition reuses the same matrix products as
// the rest of the GPU path, and it is gradient-checked against the CPU LSTM.
//
// The gate nonlinearities and cell updates are elementwise and run on the host, with one
// upload and one download per time step. The input projections and the weight gradients
// are single GEMMs over all time steps.

// lstm_cuda_eligible reports whether LSTM forward and backward may run on cuBLAS.
pub fn lstm_cuda_eligible() bool {
	return autograd_cuda.cuda_backward_enabled()
}

// lstm_backward_cuda_f64 mirrors lstm_backward_single_with_final_state on cuBLAS and
// returns the same seven gradients: d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_hidden0,
// d_cell0.
fn lstm_backward_cuda_f64(input &vtl.Tensor[f64], hidden0 &vtl.Tensor[f64], cell0 &vtl.Tensor[f64], w_ih &vtl.Tensor[f64], w_hh &vtl.Tensor[f64], b_ih &vtl.Tensor[f64], b_hh &vtl.Tensor[f64], grad_output &vtl.Tensor[f64], grad_final_hidden &vtl.Tensor[f64], grad_final_cell &vtl.Tensor[f64]) ![]&vtl.Tensor[f64] {
	if input.shape.len != 3 || hidden0.shape.len != 2 || cell0.shape != hidden0.shape {
		return error('lstm cuBLAS backward: expected input [sequence, batch, features] and matching [batch, hidden] states')
	}
	seq_len := input.shape[0]
	batch := input.shape[1]
	input_size := input.shape[2]
	hidden := hidden0.shape[1]
	four_h := 4 * hidden
	if hidden0.shape[0] != batch || w_ih.shape != [four_h, input_size]
		|| w_hh.shape != [four_h, hidden] || b_ih.size() != four_h || b_hh.size() != four_h {
		return error('lstm cuBLAS backward: incompatible input, state, weight, or bias dimensions')
	}
	if grad_output.shape != [seq_len, batch, hidden] {
		return error('lstm cuBLAS backward: grad_output shape must match output')
	}
	if grad_final_hidden.shape != hidden0.shape || grad_final_cell.shape != cell0.shape {
		return error('lstm cuBLAS backward: final-state gradient shapes must match initial states')
	}
	tokens := seq_len * batch

	x_values := input.to_array()
	wih_values := w_ih.to_array()
	whh_values := w_hh.to_array()
	bih_values := b_ih.to_array()
	bhh_values := b_hh.to_array()

	mut dev := cuda.get_default_device()!
	defer {
		dev.release() or {}
	}
	handle := dev.cublas

	d_x := attn_upload(x_values)!
	defer {
		C.cudaFree(d_x)
	}
	d_wih := attn_upload(wih_values)!
	defer {
		C.cudaFree(d_wih)
	}
	d_whh := attn_upload(whh_values)!
	defer {
		C.cudaFree(d_whh)
	}
	d_pre := device_alloc_bytes(tokens * four_h * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_pre)
	}
	d_hprev := device_alloc_bytes(batch * hidden * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_hprev)
	}

	// Input projections for every time step: pre[TB, 4H] = X[TB, I] * W_ih^T.
	attn_gemm(handle, false, true, tokens, four_h, input_size, 1.0, d_x, input_size, d_wih,
		input_size, 0.0, d_pre, four_h)!

	// Forward recurrence. Each step adds H_{t-1} * W_hh^T on the device, then the host
	// applies the bias, the gate nonlinearities, and the cell and hidden updates.
	mut hs := [][]f64{len: seq_len + 1}
	mut cs := [][]f64{len: seq_len + 1}
	mut gate_is := [][]f64{len: seq_len}
	mut gate_fs := [][]f64{len: seq_len}
	mut gate_gs := [][]f64{len: seq_len}
	mut gate_os := [][]f64{len: seq_len}
	hidden0_values := hidden0.to_array()
	cell0_values := cell0.to_array()
	hs[0] = hidden0_values.clone()
	cs[0] = cell0_values.clone()
	mut pre_step := []f64{len: batch * four_h}
	for t in 0 .. seq_len {
		hs[t + 1] = []f64{len: batch * hidden}
		cs[t + 1] = []f64{len: batch * hidden}
		gate_is[t] = []f64{len: batch * hidden}
		gate_fs[t] = []f64{len: batch * hidden}
		gate_gs[t] = []f64{len: batch * hidden}
		gate_os[t] = []f64{len: batch * hidden}
		device_upload_f64(d_hprev, hs[t])!
		step_pre := attn_offset(d_pre, t * batch * four_h)
		attn_gemm(handle, false, true, batch, four_h, hidden, 1.0, d_hprev, hidden, d_whh, hidden,
			1.0, step_pre, four_h)!
		device_download_f64(step_pre, mut pre_step)!
		for b in 0 .. batch {
			for j in 0 .. hidden {
				mut gates := [f64(0), f64(0), f64(0), f64(0)]
				for gate in 0 .. 4 {
					row := gate * hidden + j
					gates[gate] = pre_step[b * four_h + row] + bih_values[row] + bhh_values[row]
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

	// Backward through time. The gate gradients dG_t are computed on the host, which
	// reproduces the CPU BPTT loop. The recurrent gradient dH_{t-1} = dG_t * W_hh is a GEMM.
	grad_out_values := grad_output.to_array()
	grad_fh := grad_final_hidden.to_array()
	grad_fc := grad_final_cell.to_array()
	mut dg_all := []f64{len: tokens * four_h}
	mut d_h_next := grad_fh.clone()
	mut d_c_next := grad_fc.clone()
	d_dg := device_alloc_bytes(batch * four_h * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dg)
	}
	d_dh := device_alloc_bytes(batch * hidden * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dh)
	}
	mut dh_step := []f64{len: batch * hidden}
	for rev in 0 .. seq_len {
		t := seq_len - 1 - rev
		mut d_c_prev := []f64{len: batch * hidden}
		for b in 0 .. batch {
			for j in 0 .. hidden {
				idx := b * hidden + j
				d_h := grad_out_values[t * batch * hidden + idx] + d_h_next[idx]
				tanh_c := math.tanh(cs[t + 1][idx])
				d_c := d_c_next[idx] + d_h * gate_os[t][idx] * (1 - tanh_c * tanh_c)
				i := gate_is[t][idx]
				f := gate_fs[t][idx]
				g := gate_gs[t][idx]
				o := gate_os[t][idx]
				d_gates := [d_c * g * i * (1 - i), d_c * cs[t][idx] * f * (1 - f),
					d_c * i * (1 - g * g), d_h * tanh_c * o * (1 - o)]
				d_c_prev[idx] = d_c * f
				for gate in 0 .. 4 {
					row := gate * hidden + j
					dg_all[(t * batch + b) * four_h + row] = d_gates[gate]
				}
			}
		}
		// dH_{t-1} = dG_t * W_hh on the device.
		device_upload_f64(d_dg, dg_all[t * batch * four_h..(t + 1) * batch * four_h])!
		attn_gemm(handle, false, false, batch, hidden, four_h, 1.0, d_dg, four_h, d_whh, hidden,
			0.0, d_dh, hidden)!
		device_download_f64(d_dh, mut dh_step)!
		d_h_next = dh_step.clone()
		d_c_next = d_c_prev.clone()
	}

	// Weight and input gradients over all time steps at once.
	d_dgall := attn_upload(dg_all)!
	defer {
		C.cudaFree(d_dgall)
	}
	mut hprev_all := []f64{len: tokens * hidden}
	for t in 0 .. seq_len {
		for i in 0 .. batch * hidden {
			hprev_all[t * batch * hidden + i] = hs[t][i]
		}
	}
	d_hprev_all := attn_upload(hprev_all)!
	defer {
		C.cudaFree(d_hprev_all)
	}
	d_dx := device_alloc_bytes(tokens * input_size * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dx)
	}
	d_dwih := device_alloc_bytes(four_h * input_size * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dwih)
	}
	d_dwhh := device_alloc_bytes(four_h * hidden * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dwhh)
	}
	// dX[TB, I] = dG[TB, 4H] * W_ih[4H, I]
	attn_gemm(handle, false, false, tokens, input_size, four_h, 1.0, d_dgall, four_h, d_wih,
		input_size, 0.0, d_dx, input_size)!
	// dW_ih[4H, I] = dG^T[4H, TB] * X[TB, I]
	attn_gemm(handle, true, false, four_h, input_size, tokens, 1.0, d_dgall, four_h, d_x,
		input_size, 0.0, d_dwih, input_size)!
	// dW_hh[4H, H] = dG^T[4H, TB] * H_prev[TB, H]
	attn_gemm(handle, true, false, four_h, hidden, tokens, 1.0, d_dgall, four_h, d_hprev_all,
		hidden, 0.0, d_dwhh, hidden)!

	mut dx_values := []f64{len: tokens * input_size}
	device_download_f64(d_dx, mut dx_values)!
	mut dwih_values := []f64{len: four_h * input_size}
	device_download_f64(d_dwih, mut dwih_values)!
	mut dwhh_values := []f64{len: four_h * hidden}
	device_download_f64(d_dwhh, mut dwhh_values)!

	// Bias gradients are column sums of dG over all rows; both biases get the same sum.
	mut db_values := []f64{len: four_h}
	for row_index in 0 .. tokens {
		for c in 0 .. four_h {
			db_values[c] += dg_all[row_index * four_h + c]
		}
	}

	return [
		vtl.from_array(dx_values, input.shape)!,
		vtl.from_array(dwih_values, w_ih.shape)!,
		vtl.from_array(dwhh_values, w_hh.shape)!,
		vtl.from_array(db_values.clone(), b_ih.shape)!,
		vtl.from_array(db_values, b_hh.shape)!,
		vtl.from_array(d_h_next, hidden0.shape)!,
		vtl.from_array(d_c_next, cell0.shape)!,
	]
}
