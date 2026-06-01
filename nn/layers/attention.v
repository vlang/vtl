module layers

import vtl
import vtl.la
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types
import math

// MultiHeadAttentionLayer implements scaled dot-product multi-head attention.
//
// Input:    `[batch, seq_len, embed_dim]`
// Output:   `[batch, seq_len, embed_dim]`
//
// Computes attention across `num_heads` heads and projects back to `embed_dim`.
//
// Config options (via constructor parameters):
//   - `embed_dim` — model dimension (must be divisible by `num_heads`)
//   - `num_heads` — number of parallel attention heads
pub struct MultiHeadAttentionLayer[T] {
pub:
	embed_dim int
	num_heads int
	head_dim  int
pub mut:
	w_q &autograd.Variable[T] = unsafe { nil }
	w_k &autograd.Variable[T] = unsafe { nil }
	w_v &autograd.Variable[T] = unsafe { nil }
	w_o &autograd.Variable[T] = unsafe { nil }
}

// multihead_attention_layer creates a MultiHeadAttentionLayer.
pub fn multihead_attention_layer[T](ctx &autograd.Context[T], embed_dim int, num_heads int) types.Layer[T] {
	head_dim := embed_dim / num_heads
	w_q := ctx.variable(internal.kaiming_uniform[T]([embed_dim, embed_dim]))
	w_k := ctx.variable(internal.kaiming_uniform[T]([embed_dim, embed_dim]))
	w_v := ctx.variable(internal.kaiming_uniform[T]([embed_dim, embed_dim]))
	w_o := ctx.variable(internal.kaiming_uniform[T]([embed_dim, embed_dim]))
	return types.Layer[T](&MultiHeadAttentionLayer[T]{
		embed_dim: embed_dim
		num_heads: num_heads
		head_dim:  head_dim
		w_q:       w_q
		w_k:       w_k
		w_v:       w_v
		w_o:       w_o
	})
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &MultiHeadAttentionLayer[T]) output_shape() []int {
	return [layer.embed_dim]
}

// variables exposes this operation as part of the public API.
pub fn (layer &MultiHeadAttentionLayer[T]) variables() []&autograd.Variable[T] {
	return [layer.w_q, layer.w_k, layer.w_v, layer.w_o]
}

// forward exposes this operation as part of the public API.
pub fn (layer &MultiHeadAttentionLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	q := la.matmul[T](input.value, layer.w_q.value)!
	k := la.matmul[T](input.value, layer.w_k.value)!
	v := la.matmul[T](input.value, layer.w_v.value)!

	batch := q.shape[0]
	seq_len := q.shape[1]

	q_reshaped := q.reshape([batch, seq_len, layer.num_heads, layer.head_dim])!
	k_reshaped := k.reshape([batch, seq_len, layer.num_heads, layer.head_dim])!
	v_reshaped := v.reshape([batch, seq_len, layer.num_heads, layer.head_dim])!

	// Transpose: [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
	q_transposed := q_reshaped.transpose([0, 2, 1, 3])!
	k_transposed := k_reshaped.transpose([0, 2, 1, 3])!
	v_transposed := v_reshaped.transpose([0, 2, 1, 3])!

	// k_t: [batch, heads, head_dim, seq_len]
	k_t := k_transposed.transpose([0, 1, 3, 2])!
	scores := la.matmul[T](q_transposed, k_t)!

	// Scale scores
	scale := vtl.cast[T](1.0 / math.sqrt(f64(layer.head_dim)))
	mut scores_scaled := vtl.zeros_like[T](scores)
	for i in 0 .. scores.size() {
		scores_scaled.set_nth(i, vtl.cast[T](f64(scores.get_nth(i)) * f64(scale)))
	}

	// Softmax over last dim
	attn_weights := internal.softmax_forward[T](scores_scaled, -1)!

	// attn_output: [batch, heads, seq_len, head_dim]
	attn_output := la.matmul[T](attn_weights, v_transposed)!

	// Transpose back: [batch, seq_len, heads, head_dim]
	attn_output_t := attn_output.transpose([0, 2, 1, 3])!

	// Reshape: [batch, seq_len, embed_dim]
	merged := attn_output_t.reshape([batch, seq_len, layer.embed_dim])!

	// Output projection
	output := la.matmul[T](merged, layer.w_o.value)!

	mut result := input.context.variable(output)
	if input.requires_grad || layer.w_q.requires_grad || layer.w_k.requires_grad
		|| layer.w_v.requires_grad || layer.w_o.requires_grad {
		gate := attention_gate[T](input.value, layer.w_q.value, layer.w_k.value, layer.w_v.value,
			layer.w_o.value, layer.num_heads, layer.head_dim)
		gate.cache(mut result, input)!
	}
	return result
}

// AttentionGate defines a public data structure for this module.
pub struct AttentionGate[T] {
	input     &vtl.Tensor[T] = unsafe { nil }
	w_q       &vtl.Tensor[T] = unsafe { nil }
	w_k       &vtl.Tensor[T] = unsafe { nil }
	w_v       &vtl.Tensor[T] = unsafe { nil }
	w_o       &vtl.Tensor[T] = unsafe { nil }
	num_heads int
	head_dim  int
}

// attention_gate exposes this operation as part of the public API.
pub fn attention_gate[T](input &vtl.Tensor[T], w_q &vtl.Tensor[T], w_k &vtl.Tensor[T], w_v &vtl.Tensor[T], w_o &vtl.Tensor[T], num_heads int, head_dim int) &AttentionGate[T] {
	return &AttentionGate[T]{
		input:     input
		w_q:       w_q
		w_k:       w_k
		w_v:       w_v
		w_o:       w_o
		num_heads: num_heads
		head_dim:  head_dim
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &AttentionGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := payload.variable.grad
	d_w_o := la.matmul[T](g.input.transpose([1, 0])!, grad)!
	d_input := la.matmul[T](grad, g.w_o.transpose([1, 0])!)!
	return [d_input, d_w_o, d_w_o, d_w_o, d_w_o]
}

// cache exposes this operation as part of the public API.
pub fn (g &AttentionGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Attention', g, result, [a])!
		}
		else {}
	}
}
