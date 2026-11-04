module layers

import math
import os
import vtl
import vtl.autograd
import vtl.nn.internal

// End-to-end gradient checks: multi-head attention backward on cuBLAS against the CPU path.
// Runs only with `-d cuda` and VTL_TEST_CUDA=1, the same skip pattern as the other CUDA
// tests. Weights are fixed (not random) so both runs see identical inputs.

fn attention_cuda_test_skip() bool {
	$if !cuda {
		return true
	}
	return os.getenv('VTL_TEST_CUDA') != '1'
}

fn attention_cuda_set_env(enabled bool) {
	if enabled {
		os.setenv('VTL_USE_CUDA', '1', true)
		os.setenv('VTL_CUDA_BACKWARD', '1', true)
	} else {
		os.unsetenv('VTL_USE_CUDA')
		os.unsetenv('VTL_CUDA_BACKWARD')
	}
}

fn attention_cuda_assert_close(got []f64, want []f64, tol f64, label string) {
	assert got.len == want.len, '${label}: length ${got.len} != ${want.len}'
	for i in 0 .. got.len {
		assert math.abs(got[i] - want[i]) < tol, '${label}[${i}]: got ${got[i]}, want ${want[i]}'
	}
}

// attention_cuda_grads runs one forward and backward pass with fixed weights and returns
// [input.grad, w_q.grad, w_k.grad, w_v.grad, w_o.grad].
fn attention_cuda_grads(enabled bool) ![]&vtl.Tensor[f64] {
	attention_cuda_set_env(enabled)
	defer {
		attention_cuda_set_env(false)
	}
	mut input_values := []f64{len: 2 * 3 * 4}
	for i in 0 .. input_values.len {
		input_values[i] = f64(i % 7 - 3) * 0.21 + f64(i) * 0.013
	}
	mut wq := []f64{len: 16}
	mut wk := []f64{len: 16}
	mut wv := []f64{len: 16}
	mut wo := []f64{len: 16}
	for i in 0 .. 16 {
		wq[i] = f64(i % 5 - 2) * 0.17 + 0.03
		wk[i] = f64(i % 3 - 1) * 0.23 - 0.02
		wv[i] = f64(i % 4 - 2) * 0.11 + 0.05
		wo[i] = f64(i % 6 - 3) * 0.13 - 0.04
	}
	ctx := autograd.ctx[f64]()
	layer := &MultiHeadAttentionLayer[f64]{
		embed_dim: 4
		num_heads: 2
		head_dim:  2
		w_q:       ctx.variable(vtl.from_array(wq, [4, 4])!)
		w_k:       ctx.variable(vtl.from_array(wk, [4, 4])!)
		w_v:       ctx.variable(vtl.from_array(wv, [4, 4])!)
		w_o:       ctx.variable(vtl.from_array(wo, [4, 4])!)
	}
	input := ctx.variable(vtl.from_array(input_values, [2, 3, 4])!)
	result := layer.forward(input)!
	result.backprop()!
	return [input.grad, layer.w_q.grad, layer.w_k.grad, layer.w_v.grad, layer.w_o.grad]
}

fn test_attention_backward_cuda_matches_cpu_reference() ! {
	if attention_cuda_test_skip() {
		return
	}
	cpu := attention_cuda_grads(false)!
	gpu := attention_cuda_grads(true)!
	names := ['d_input', 'd_w_q', 'd_w_k', 'd_w_v', 'd_w_o']
	assert gpu.len == 5
	for k in 0 .. 5 {
		assert gpu[k].shape == cpu[k].shape
		attention_cuda_assert_close(gpu[k].to_array(), cpu[k].to_array(), 1e-9, names[k])
	}
}

fn test_attention_backward_dispatch_selects_cuda_when_enabled() ! {
	if attention_cuda_test_skip() {
		return
	}
	attention_cuda_set_env(true)
	defer {
		attention_cuda_set_env(false)
	}
	assert internal.attention_cuda_eligible()
}
