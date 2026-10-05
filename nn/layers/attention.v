module layers

import vtl
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
		gate.q = q
		gate.k = k
		gate.v = v
		gate.attn_weights = attn_weights
		gate.merged = merged
		gate.cache(mut result, input, layer.w_q, layer.w_k, layer.w_v, layer.w_o)!
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
	input        &vtl.Tensor[T] = unsafe { nil }
	w_q          &vtl.Tensor[T] = unsafe { nil }
	w_k          &vtl.Tensor[T] = unsafe { nil }
	w_v          &vtl.Tensor[T] = unsafe { nil }
	w_o          &vtl.Tensor[T] = unsafe { nil }
	q            &vtl.Tensor[T] = unsafe { nil }
	k            &vtl.Tensor[T] = unsafe { nil }
	v            &vtl.Tensor[T] = unsafe { nil }
	attn_weights &vtl.Tensor[T] = unsafe { nil }
	merged       &vtl.Tensor[T] = unsafe { nil }
	num_heads    int
	head_dim     int
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
	batch := g.input.shape[0]
	seq_len := g.input.shape[1]
	embed_dim := g.input.shape[2]
	mut d_input := vtl.zeros[T](g.input.shape)
	mut d_w_q := vtl.zeros[T](g.w_q.shape)
	mut d_w_k := vtl.zeros[T](g.w_k.shape)
	mut d_w_v := vtl.zeros[T](g.w_v.shape)
	mut d_w_o := vtl.zeros[T](g.w_o.shape)
	mut d_q := vtl.zeros[T](g.q.shape)
	mut d_k := vtl.zeros[T](g.k.shape)
	mut d_v := vtl.zeros[T](g.v.shape)
	mut d_merged := vtl.zeros[T](g.merged.shape)
	mut d_attn := vtl.zeros[T](g.attn_weights.shape)
	mut d_scores := vtl.zeros[T](g.attn_weights.shape)
	mut d_input_q := vtl.zeros[T](g.input.shape)
	mut d_input_k := vtl.zeros[T](g.input.shape)
	mut d_input_v := vtl.zeros[T](g.input.shape)

	// Backpropagate the output projection: merged @ W_o.
	for b in 0 .. batch {
		for s in 0 .. seq_len {
			for i in 0 .. embed_dim {
				mut sum := f64(0)
				for o in 0 .. embed_dim {
					dout := f64(grad.get([b, s, o]))
					sum += dout * f64(g.w_o.get([i, o]))
					d_w_o.set([i, o], vtl.cast[T](f64(d_w_o.get([i, o])) + f64(g.merged.get([
						b,
						s,
						i,
					])) * dout))
				}
				d_merged.set([b, s, i], vtl.cast[T](sum))
			}
		}
	}

	// Backpropagate attention output, softmax, and scaled QK^T scores.
	scale := 1.0 / math.sqrt(f64(g.head_dim))
	for b in 0 .. batch {
		for h in 0 .. g.num_heads {
			for i in 0 .. seq_len {
				mut softmax_dot := f64(0)
				for j in 0 .. seq_len {
					mut d_weight := f64(0)
					for d in 0 .. g.head_dim {
						feature := h * g.head_dim + d
						d_context := f64(d_merged.get([b, i, feature]))
						d_weight += d_context * f64(g.v.get([b, j, feature]))
						d_v.set([b, j, feature], vtl.cast[T](f64(d_v.get([b, j, feature])) +
							f64(g.attn_weights.get([b, h, i, j])) * d_context))
					}
					d_attn.set([b, h, i, j], vtl.cast[T](d_weight))
					softmax_dot += d_weight * f64(g.attn_weights.get([b, h, i, j]))
				}
				for j in 0 .. seq_len {
					a := f64(g.attn_weights.get([b, h, i, j]))
					d_scores.set([b, h, i, j], vtl.cast[T](a * (f64(d_attn.get([b, h, i, j])) -
						softmax_dot)))
				}
			}
			for i in 0 .. seq_len {
				for d in 0 .. g.head_dim {
					feature := h * g.head_dim + d
					mut q_grad := f64(0)
					for j in 0 .. seq_len {
						q_grad += f64(d_scores.get([b, h, i, j])) * f64(g.k.get([b, j, feature])) * scale
					}
					d_q.set([b, i, feature], vtl.cast[T](q_grad))
				}
			}
			for j in 0 .. seq_len {
				for d in 0 .. g.head_dim {
					feature := h * g.head_dim + d
					mut k_grad := f64(0)
					for i in 0 .. seq_len {
						k_grad += f64(d_scores.get([b, h, i, j])) * f64(g.q.get([b, i, feature])) * scale
					}
					d_k.set([b, j, feature], vtl.cast[T](k_grad))
				}
			}
		}
	}

	// Gradients for the Q/K/V projections and their contributions to the input.
	for b in 0 .. batch {
		for s in 0 .. seq_len {
			for i in 0 .. embed_dim {
				for o in 0 .. embed_dim {
					x := f64(g.input.get([b, s, i]))
					dq := f64(d_q.get([b, s, o]))
					dk := f64(d_k.get([b, s, o]))
					dv := f64(d_v.get([b, s, o]))
					d_w_q.set([i, o], vtl.cast[T](f64(d_w_q.get([i, o])) + x * dq))
					d_w_k.set([i, o], vtl.cast[T](f64(d_w_k.get([i, o])) + x * dk))
					d_w_v.set([i, o], vtl.cast[T](f64(d_w_v.get([i, o])) + x * dv))
					d_input_q.set([b, s, i], vtl.cast[T](f64(d_input_q.get([b, s, i])) + dq * f64(g.w_q.get([
						i,
						o,
					]))))
					d_input_k.set([b, s, i], vtl.cast[T](f64(d_input_k.get([b, s, i])) + dk * f64(g.w_k.get([
						i,
						o,
					]))))
					d_input_v.set([b, s, i], vtl.cast[T](f64(d_input_v.get([b, s, i])) + dv * f64(g.w_v.get([
						i,
						o,
					]))))
				}
				d_input.set([b, s, i], vtl.cast[T](f64(d_input_q.get([b, s, i])) +
					f64(d_input_k.get([b, s, i])) + f64(d_input_v.get([b, s, i]))))
			}
		}
	}
	return [d_input, d_w_q, d_w_k, d_w_v, d_w_o]
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
			mut parents := []&autograd.Variable[T]{cap: args.len}
			for arg in args {
				match arg {
					autograd.Variable[T] {
						parents << arg
					}
					else {}
				}
			}
			autograd.register[T]('Attention', voidptr(g), attention_gate_backward_dispatch[T],
				result, parents)!
		}
		else {}
	}
}
