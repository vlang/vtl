module layers

import vtl
import vtl.la
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types
import math

// MultiHeadAttention layer.
// input: [batch, seq_len, embed_dim]
// Returns: [batch, seq_len, embed_dim] (same as input, projection back to embed_dim)
pub struct MultiHeadAttentionLayer[T] {
	embed_dim       int
	num_heads       int
	head_dim        int
pub mut:
	w_q             &autograd.Variable[T] = unsafe { nil }
	w_k             &autograd.Variable[T] = unsafe { nil }
	w_v             &autograd.Variable[T] = unsafe { nil }
	w_o             &autograd.Variable[T] = unsafe { nil }
}

pub fn multihead_attention_layer[T](ctx &autograd.Context[T], embed_dim int, num_heads int) types.Layer[T] {
	head_dim := embed_dim / num_heads
	w_q := ctx.variable(internal.kaiming_uniform[T]([embed_dim, embed_dim]))
	w_k := ctx.variable(internal.kaiming_uniform[T]([embed_dim, embed_dim]))
	w_v := ctx.variable(internal.kaiming_uniform[T]([embed_dim, embed_dim]))
	w_o := ctx.variable(internal.kaiming_uniform[T]([embed_dim, embed_dim]))
	return types.Layer[T](&MultiHeadAttentionLayer[T]{
		embed_dim: embed_dim, num_heads: num_heads, head_dim: head_dim,
		w_q: w_q, w_k: w_k, w_v: w_v, w_o: w_o
	})
}

pub fn (layer &MultiHeadAttentionLayer[T]) output_shape() []int { return []int{layer.embed_dim} }
pub fn (layer &MultiHeadAttentionLayer[T]) variables() []&autograd.Variable[T] { return [layer.w_q, layer.w_k, layer.w_v, layer.w_o] }

pub fn (layer &MultiHeadAttentionLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	q := la.matmul[T](input.value, layer.w_q.value)!
	k := la.matmul[T](input.value, layer.w_k.value)!
	v := la.matmul[T](input.value, layer.w_v.value)!

	batch := q.shape[0]
	seq_len := q.shape[1]
	q_reshaped := q.reshape([batch, seq_len, layer.num_heads, layer.head_dim])!
	k_reshaped := k.reshape([batch, seq_len, layer.num_heads, layer.head_dim])!
	v_reshaped := v.reshape([batch, seq_len, layer.num_heads, layer.head_dim])!

	q_transposed := q_reshaped.transpose(1, 2)!
	k_transposed := k_reshaped.transpose(1, 2)!
	v_transposed := v_reshaped.transpose(1, 2)!

	k_t := k_transposed.t()!
	scores := la.matmul[T](q_transposed, k_t)!
	scores := scores.map(fn [layer] [T](val T, i []int) T {
		return val / vtl.cast[T](math.sqrt(f64(layer.head_dim)))
	})!

	attn_weights := internal.softmax[T](scores, 3)!

	attn_output := la.matmul[T](attn_weights, v_transposed)!

	attn_output_transposed := attn_output.transpose(1, 2)!

	output := attn_output_transposed.reshape([batch, seq_len, layer.embed_dim])!

	output = la.matmul[T](output, layer.w_o.value)!

	mut result := input.context.variable(output)
	if input.requires_grad || layer.w_q.requires_grad || layer.w_k.requires_grad || layer.w_v.requires_grad || layer.w_o.requires_grad {
		gate := attention_gate[T](input.value, layer.w_q.value, layer.w_k.value, layer.w_v.value, layer.w_o.value, layer.num_heads, layer.head_dim)
		gate.cache(mut result, input)!
	}
	return result
}

pub struct AttentionGate[T] {
	input   &vtl.Tensor[T] = unsafe { nil }
	w_q     &vtl.Tensor[T] = unsafe { nil }
	w_k     &vtl.Tensor[T] = unsafe { nil }
	w_v     &vtl.Tensor[T] = unsafe { nil }
	w_o     &vtl.Tensor[T] = unsafe { nil }
	num_heads int
	head_dim   int
}

pub fn attention_gate[T](input &vtl.Tensor[T], w_q &vtl.Tensor[T], w_k &vtl.Tensor[T], w_v &vtl.Tensor[T], w_o &vtl.Tensor[T], num_heads int, head_dim int) &AttentionGate[T] {
	return &AttentionGate[T]{input: input, w_q: w_q, w_k: w_k, w_v: w_v, w_o: w_o, num_heads: num_heads, head_dim: head_dim}
}

pub fn (g &AttentionGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := payload.variable.grad
	d_w_o := la.matmul[T](g.input.t()!, grad)!
	d_input := la.matmul[T](grad, g.w_o.t()!)!
	return [d_input, d_w_o, d_w_o, d_w_o, d_w_o]
}

pub fn (g &AttentionGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	match args[0] {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Attention', g, result, [args[0]])!
		}
		else {}
	}
}
