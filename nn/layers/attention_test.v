module layers

import vtl
import vtl.autograd
import math

fn test_multihead_attention_forward_supports_batched_sequences() ! {
	ctx := autograd.ctx[f64]()
	layer := multihead_attention_layer[f64](ctx, 4, 2)
	input := ctx.variable(vtl.from_array([
		f64(1),
		0.5,
		-0.5,
		0.25,
		0.25,
		-1,
		0.75,
		1,
		-0.5,
		0.25,
		1,
		-0.75,
		1.5,
		-0.25,
		0.5,
		0.125,
		-0.75,
		0.5,
		0.25,
		1.25,
		0.5,
		1.25,
		-0.5,
		0.75,
	], [2, 3, 4])!)

	output := layer.forward(input)!
	assert output.value.shape == [2, 3, 4]
	for i in 0 .. output.value.size() {
		value := output.value.get_nth(i)
		assert value == value, 'attention output contains NaN at flat index ${i}'
	}
}

fn test_multihead_attention_rejects_zero_heads_without_panicking() {
	ctx := autograd.ctx[f64]()
	layer := multihead_attention_layer[f64](ctx, 4, 0)
	input := ctx.variable(vtl.ones[f64]([1, 2, 4]))
	if _ := layer.forward(input) {
		assert false, 'attention must reject zero heads'
	} else {
		assert true
	}
}

fn test_multihead_attention_backward_matches_finite_differences() ! {
	input_values := [f64(0.2), -0.4, 0.7, 0.3]
	wq_values := [f64(0.1), 0.2, -0.3, 0.4]
	wk_values := [f64(-0.2), 0.3, 0.5, -0.1]
	wv_values := [f64(0.6), -0.2, 0.1, 0.3]
	wo_values := [f64(0.2), 0.4, -0.1, 0.5]
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array(input_values, [1, 2, 2])!)
	w_q := ctx.variable(vtl.from_array(wq_values, [2, 2])!)
	w_k := ctx.variable(vtl.from_array(wk_values, [2, 2])!)
	w_v := ctx.variable(vtl.from_array(wv_values, [2, 2])!)
	w_o := ctx.variable(vtl.from_array(wo_values, [2, 2])!)
	layer := &MultiHeadAttentionLayer[f64]{
		embed_dim: 2
		num_heads: 1
		head_dim:  2
		w_q:       w_q
		w_k:       w_k
		w_v:       w_v
		w_o:       w_o
	}
	result := layer.forward(input)!
	result.backprop()!

	epsilon := f64(1e-6)
	for i in 0 .. input_values.len {
		mut plus := input_values.clone()
		mut minus := input_values.clone()
		plus[i] += epsilon
		minus[i] -= epsilon
		numeric := (attention_objective(plus, wq_values, wk_values, wv_values, wo_values)! - attention_objective(minus,
			wq_values, wk_values, wv_values, wo_values)!) / (2 * epsilon)
		assert math.abs(numeric - f64(input.grad.get_nth(i))) < 1e-5
	}
	weight_values := [wq_values, wk_values, wv_values, wo_values]
	weight_gradients := [w_q.grad, w_k.grad, w_v.grad, w_o.grad]
	for weight_index in 0 .. weight_values.len {
		values := weight_values[weight_index]
		gradient := weight_gradients[weight_index]
		for i in 0 .. values.len {
			mut plus := values.clone()
			mut minus := values.clone()
			plus[i] += epsilon
			minus[i] -= epsilon
			mut plus_weights := [wq_values.clone(), wk_values.clone(), wv_values.clone(),
				wo_values.clone()]
			mut minus_weights := [wq_values.clone(), wk_values.clone(), wv_values.clone(),
				wo_values.clone()]
			plus_weights[weight_index] = plus
			minus_weights[weight_index] = minus
			numeric := (attention_objective(input_values, plus_weights[0], plus_weights[1], plus_weights[2], plus_weights[3])! -
				attention_objective(input_values, minus_weights[0], minus_weights[1], minus_weights[2],
					minus_weights[3])!) / (2 * epsilon)
			assert math.abs(numeric - f64(gradient.get_nth(i))) < 1e-5
		}
	}
}

fn attention_objective(input_values []f64, wq_values []f64, wk_values []f64, wv_values []f64, wo_values []f64) !f64 {
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array(input_values, [1, 2, 2])!, requires_grad: false)
	w_q := ctx.variable(vtl.from_array(wq_values, [2, 2])!, requires_grad: false)
	w_k := ctx.variable(vtl.from_array(wk_values, [2, 2])!, requires_grad: false)
	w_v := ctx.variable(vtl.from_array(wv_values, [2, 2])!, requires_grad: false)
	w_o := ctx.variable(vtl.from_array(wo_values, [2, 2])!, requires_grad: false)
	layer := &MultiHeadAttentionLayer[f64]{
		embed_dim: 2
		num_heads: 1
		head_dim:  2
		w_q:       w_q
		w_k:       w_k
		w_v:       w_v
		w_o:       w_o
	}
	output := layer.forward(input)!
	mut total := f64(0)
	for value in output.value.to_array() {
		total += value
	}
	return total
}
