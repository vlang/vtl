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
	layer := &MultiHeadAttentionLayer[T]{
		embed_dim: embed_dim
		num_heads: num_heads
		head_dim:  head_dim
		w_q:       w_q
		w_k:       w_k
		w_v:       w_v
		w_o:       w_o
	}
	return types.layer[T](voidptr(layer), multi_head_attention_layer_output_shape_dispatch[T],
		multi_head_attention_layer_variables_dispatch[T],
		multi_head_attention_layer_forward_dispatch[T])
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
	if input.value.shape.len != 3 {
		return error('multihead attention expects [batch, seq_len, embed_dim] input')
	}
	if layer.num_heads <= 0 || layer.embed_dim <= 0 || layer.embed_dim % layer.num_heads != 0
		|| layer.head_dim * layer.num_heads != layer.embed_dim {
		return error('embed_dim must be positive and divisible by num_heads')
	}
	batch := input.value.shape[0]
	seq_len := input.value.shape[1]
	if input.value.shape[2] != layer.embed_dim {
		return error('input embed_dim ${input.value.shape[2]} does not match layer embed_dim ${layer.embed_dim}')
	}
	// la.matmul currently accepts rank-2 matrices only. Project each token with
	// explicit loops so attention supports its documented batched rank-3 input.
	q := attention_project[T](input.value, layer.w_q.value, batch, seq_len, layer.embed_dim)
	k := attention_project[T](input.value, layer.w_k.value, batch, seq_len, layer.embed_dim)
	v := attention_project[T](input.value, layer.w_v.value, batch, seq_len, layer.embed_dim)

	mut scores := vtl.zeros[T]([batch, layer.num_heads, seq_len, seq_len])
	scale := 1.0 / math.sqrt(f64(layer.head_dim))
	for b in 0 .. batch {
		for h in 0 .. layer.num_heads {
			for i in 0 .. seq_len {
				for j in 0 .. seq_len {
					mut dot := f64(0)
					for d in 0 .. layer.head_dim {
						q_idx := h * layer.head_dim + d
						k_idx := h * layer.head_dim + d
						dot += f64(q.get([b, i, q_idx])) * f64(k.get([b, j, k_idx]))
					}
					scores.set([b, h, i, j], vtl.cast[T](dot * scale))
				}
			}
		}
	}
	attn_weights := internal.softmax_forward[T](scores, -1)!

	mut merged := vtl.zeros[T]([batch, seq_len, layer.embed_dim])
	for b in 0 .. batch {
		for i in 0 .. seq_len {
			for h in 0 .. layer.num_heads {
				for d in 0 .. layer.head_dim {
					mut sum := f64(0)
					for j in 0 .. seq_len {
						v_idx := h * layer.head_dim + d
						sum += f64(attn_weights.get([b, h, i, j])) * f64(v.get([b, j, v_idx]))
					}
					merged.set([b, i, h * layer.head_dim + d], vtl.cast[T](sum))
				}
			}
		}
	}

	output := attention_project[T](merged, layer.w_o.value, batch, seq_len, layer.embed_dim)

	mut result := input.context.variable(output)
	if input.requires_grad || layer.w_q.requires_grad || layer.w_k.requires_grad
		|| layer.w_v.requires_grad || layer.w_o.requires_grad {
		gate := attention_gate[T](input.value, layer.w_q.value, layer.w_k.value, layer.w_v.value,
			layer.w_o.value, layer.num_heads, layer.head_dim)
		gate.cache(mut result, input)!
	}
	return result
}

fn attention_project[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], batch int, seq_len int, embed_dim int) &vtl.Tensor[T] {
	mut output := vtl.zeros[T]([batch, seq_len, embed_dim])
	for b in 0 .. batch {
		for s in 0 .. seq_len {
			for out_dim in 0 .. embed_dim {
				mut sum := f64(0)
				for in_dim in 0 .. embed_dim {
					sum += f64(input.get([b, s, in_dim])) * f64(weight.get([in_dim, out_dim]))
				}
				output.set([b, s, out_dim], vtl.cast[T](sum))
			}
		}
	}
	return output
}

fn multi_head_attention_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&MultiHeadAttentionLayer[T](layer)).output_shape() }
}

fn multi_head_attention_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&MultiHeadAttentionLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn multi_head_attention_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&MultiHeadAttentionLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
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

fn attention_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&AttentionGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &AttentionGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Attention', voidptr(g), attention_gate_backward_dispatch[T],
				result, [a])!
		}
		else {}
	}
}
