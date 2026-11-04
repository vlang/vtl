module internal

import math
import os
import vtl

// Gradient checks: cuBLAS LSTM forward and BPTT backward against the CPU reference.
// Runs only with `-d cuda` and VTL_TEST_CUDA=1, the same skip pattern as the other CUDA
// tests. The CPU reference is computed with the CUDA environment cleared. Final-state
// gradients are non-zero so that their propagation is exercised as well.

fn lstm_cuda_test_skip() bool {
	$if !cuda {
		return true
	}
	return os.getenv('VTL_TEST_CUDA') != '1'
}

fn lstm_cuda_set_env(enabled bool) {
	if enabled {
		os.setenv('VTL_USE_CUDA', '1', true)
		os.setenv('VTL_CUDA_BACKWARD', '1', true)
	} else {
		os.unsetenv('VTL_USE_CUDA')
		os.unsetenv('VTL_CUDA_BACKWARD')
	}
}

fn lstm_cuda_assert_close(got []f64, want []f64, tol f64, label string) {
	assert got.len == want.len, '${label}: length ${got.len} != ${want.len}'
	for i in 0 .. got.len {
		assert math.abs(got[i] - want[i]) < tol, '${label}[${i}]: got ${got[i]}, want ${want[i]}'
	}
}

// lstm_cuda_grads runs the single-layer LSTM backward with fixed data and returns the
// seven gradients in the order documented on lstm_backward_single_with_final_state.
fn lstm_cuda_grads(enabled bool) ![]&vtl.Tensor[f64] {
	lstm_cuda_set_env(enabled)
	defer {
		lstm_cuda_set_env(false)
	}
	seq := 3
	batch := 2
	input_size := 3
	hidden := 2
	mut x := []f64{len: seq * batch * input_size}
	for i in 0 .. x.len {
		x[i] = f64(i % 7 - 3) * 0.19 + f64(i) * 0.01
	}
	mut wih := []f64{len: 4 * hidden * input_size}
	for i in 0 .. wih.len {
		wih[i] = f64(i % 5 - 2) * 0.13 - 0.02
	}
	mut whh := []f64{len: 4 * hidden * hidden}
	for i in 0 .. whh.len {
		whh[i] = f64(i % 3 - 1) * 0.21 + 0.03
	}
	mut bih := []f64{len: 4 * hidden}
	mut bhh := []f64{len: 4 * hidden}
	for i in 0 .. 4 * hidden {
		bih[i] = f64(i % 4) * 0.05 - 0.07
		bhh[i] = -f64(i % 3) * 0.04 + 0.02
	}
	mut grad_out := []f64{len: seq * batch * hidden}
	for i in 0 .. grad_out.len {
		grad_out[i] = f64(i % 5 - 2) * 0.3 + 0.1
	}
	h0 := vtl.from_array([f64(0.2), -0.1, 0.05, 0.3], [batch, hidden])!
	c0 := vtl.from_array([f64(-0.15), 0.25, 0.1, -0.2], [batch, hidden])!
	return lstm_backward_single_with_final_state[f64](vtl.from_array(x, [seq, batch, input_size])!,
		h0, c0, vtl.from_array(wih, [4 * hidden, input_size])!, vtl.from_array(whh, [
			4 * hidden,
			hidden,
		])!,
		vtl.from_array(bih, [4 * hidden])!, vtl.from_array(bhh, [4 * hidden])!,
		vtl.from_array(grad_out, [seq, batch, hidden])!, vtl.from_array([f64(0.4), -0.3, 0.2, 0.1],
			[batch, hidden])!, vtl.from_array([f64(-0.05), 0.15, 0.08, -0.12], [batch, hidden])!)!
}

fn test_lstm_backward_cuda_matches_cpu_reference() ! {
	if lstm_cuda_test_skip() {
		return
	}
	cpu := lstm_cuda_grads(false)!
	gpu := lstm_cuda_grads(true)!
	names := ['d_input', 'd_w_ih', 'd_w_hh', 'd_b_ih', 'd_b_hh', 'd_hidden0', 'd_cell0']
	assert gpu.len == 7
	for k in 0 .. 7 {
		assert gpu[k].shape == cpu[k].shape, '${names[k]} shape'
		lstm_cuda_assert_close(gpu[k].to_array(), cpu[k].to_array(), 1e-9, names[k])
	}
}

fn test_lstm_backward_dispatch_selects_cuda_when_enabled() ! {
	if lstm_cuda_test_skip() {
		return
	}
	lstm_cuda_set_env(true)
	defer {
		lstm_cuda_set_env(false)
	}
	assert lstm_cuda_eligible()
}
