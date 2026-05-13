module layers

import vtl
import vtl.la
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types
import math
import vsl.compute as vsl_compute

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

pub fn (layer &MultiHeadAttentionLayer[T]) output_shape() []int {
	return [layer.embed_dim]
}

pub fn (layer &MultiHeadAttentionLayer[T]) variables() []&autograd.Variable[T] {
	return [layer.w_q, layer.w_k, layer.w_v, layer.w_o]
}

pub fn (layer &MultiHeadAttentionLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	backend := input.context.compute_backend
	strict := input.context.compute_strict
	mut has_vulkan := false
	for b in vsl_compute.available_backends() {
		if b.str() == 'vulkan' {
			has_vulkan = true
			break
		}
	}
	if strict && backend == .vulkan && !has_vulkan {
		available := vsl_compute.available_backends().map(it.str()).join(', ')
		return error('attention: Vulkan backend unavailable in strict mode. available=[${available}]')
	}

	q := attention_project_last_dim[T](input.value, layer.w_q.value, backend, strict)!
	k := attention_project_last_dim[T](input.value, layer.w_k.value, backend, strict)!
	v := attention_project_last_dim[T](input.value, layer.w_v.value, backend, strict)!

	batch := q.shape[0]
	seq_len := q.shape[1]

	q_reshaped := q.reshape([batch, seq_len, layer.num_heads, layer.head_dim])!
	k_reshaped := k.reshape([batch, seq_len, layer.num_heads, layer.head_dim])!
	v_reshaped := v.reshape([batch, seq_len, layer.num_heads, layer.head_dim])!

	// Transpose: [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
	q_transposed := q_reshaped.transpose([0, 2, 1, 3])!
	k_transposed := k_reshaped.transpose([0, 2, 1, 3])!
	v_transposed := v_reshaped.transpose([0, 2, 1, 3])!

	mut scores := &vtl.Tensor[T](unsafe { nil })
	if backend == .vulkan || (backend == .auto && has_vulkan) {
		scores = attention_scores_vulkan[T](q_transposed, k_transposed, layer.head_dim) or {
			if strict {
				available := vsl_compute.available_backends().map(it.str()).join(', ')
				return error('attention: Vulkan score path failed in strict mode: ${err}. available=[${available}]')
			}
			k_t := k_transposed.transpose([0, 1, 3, 2])!
			attention_batched_matmul_4d[T](q_transposed, k_t)!
		}
	} else {
		if strict && backend != .cpu && backend != .auto {
			available := vsl_compute.available_backends().map(it.str()).join(', ')
			return error('attention: backend `${backend}` is not implemented for runtime GPU dispatch yet. available=[${available}]')
		}
		k_t := k_transposed.transpose([0, 1, 3, 2])!
		scores = attention_batched_matmul_4d[T](q_transposed, k_t)!
	}

	// Scale scores
	scale := vtl.cast[T](1.0 / math.sqrt(f64(layer.head_dim)))
	mut scores_scaled := vtl.zeros_like[T](scores)
	for i in 0 .. scores.size() {
		scores_scaled.set_nth(i, vtl.cast[T](f64(scores.get_nth(i)) * f64(scale)))
	}

	// Softmax over last dim
	attn_weights := internal.softmax_forward[T](scores_scaled, -1)!

	// attn_output: [batch, heads, seq_len, head_dim]
	mut attn_output := &vtl.Tensor[T](unsafe { nil })
	if backend == .vulkan || (backend == .auto && has_vulkan) {
		attn_output = attention_apply_values_vulkan[T](attn_weights, v_transposed, layer.head_dim) or {
			if strict {
				available := vsl_compute.available_backends().map(it.str()).join(', ')
				return error('attention: Vulkan value path failed in strict mode: ${err}. available=[${available}]')
			}
			attention_batched_matmul_4d[T](attn_weights, v_transposed)!
		}
	} else {
		attn_output = attention_batched_matmul_4d[T](attn_weights, v_transposed)!
	}

	// Transpose back: [batch, seq_len, heads, head_dim]
	attn_output_t := attn_output.transpose([0, 2, 1, 3])!

	// Reshape: [batch, seq_len, embed_dim]
	merged := attn_output_t.reshape([batch, seq_len, layer.embed_dim])!

	// Output projection
	output := attention_project_last_dim[T](merged, layer.w_o.value, backend, strict)!

	mut result := input.context.variable(output)
	if input.requires_grad || layer.w_q.requires_grad || layer.w_k.requires_grad
		|| layer.w_v.requires_grad || layer.w_o.requires_grad {
		gate := attention_gate[T](input.value, layer.w_q.value, layer.w_k.value, layer.w_v.value,
			layer.w_o.value, layer.num_heads, layer.head_dim)
		gate.cache(mut result, input)!
	}
	return result
}

fn attention_project_last_dim[T](x &vtl.Tensor[T], w &vtl.Tensor[T], backend vtl.Backend, strict bool) !&vtl.Tensor[T] {
	if x.shape.len == 2 {
		if w.shape.len != 2 || x.shape[1] != w.shape[0] {
			return error('attention: projection shape mismatch ${x.shape} and ${w.shape}')
		}
		if strict && backend != .cpu && backend != .auto {
			return error('attention: strict mode backend `${backend}` is not implemented for projection')
		}
		rows := x.shape[0]
		inner := x.shape[1]
		cols := w.shape[1]
		mut out := vtl.zeros[T]([rows, cols])
		for r in 0 .. rows {
			for c in 0 .. cols {
				mut sum := 0.0
				for i in 0 .. inner {
					sum += f64(x.get([r, i])) * f64(w.get([i, c]))
				}
				out.set([r, c], vtl.cast[T](sum))
			}
		}
		return out
	}
	if x.shape.len != 3 || w.shape.len != 2 {
		return error('attention: expected input [batch, seq, in] and weight [in, out], got ${x.shape} and ${w.shape}')
	}
	if x.shape[2] != w.shape[0] {
		return error('attention: projection shape mismatch ${x.shape} and ${w.shape}')
	}
	if strict && backend != .cpu && backend != .auto {
		return error('attention: strict mode backend `${backend}` is not implemented for batched projection')
	}
	batch := x.shape[0]
	seq_len := x.shape[1]
	in_dim := x.shape[2]
	out_dim := w.shape[1]
	mut out := vtl.zeros[T]([batch, seq_len, out_dim])
	for b in 0 .. batch {
		for s in 0 .. seq_len {
			for o in 0 .. out_dim {
				mut sum := 0.0
				for i in 0 .. in_dim {
					sum += f64(x.get([b, s, i])) * f64(w.get([i, o]))
				}
				out.set([b, s, o], vtl.cast[T](sum))
			}
		}
	}
	return out
}

fn attention_batched_matmul_4d[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if a.shape.len != 4 || b.shape.len != 4 {
		return error('attention: expected rank-4 tensors, got ${a.shape} and ${b.shape}')
	}
	if a.shape[0] != b.shape[0] || a.shape[1] != b.shape[1] || a.shape[3] != b.shape[2] {
		return error('attention: batched matmul shape mismatch ${a.shape} and ${b.shape}')
	}
	batch := a.shape[0]
	heads := a.shape[1]
	rows := a.shape[2]
	inner := a.shape[3]
	cols := b.shape[3]
	mut out := vtl.zeros[T]([batch, heads, rows, cols])
	for bi in 0 .. batch {
		for h in 0 .. heads {
			for r in 0 .. rows {
				for c in 0 .. cols {
					mut sum := 0.0
					for k in 0 .. inner {
						sum += f64(a.get([bi, h, r, k])) * f64(b.get([bi, h, k, c]))
					}
					out.set([bi, h, r, c], vtl.cast[T](sum))
				}
			}
		}
	}
	return out
}

pub struct AttentionGate[T] {
	input     &vtl.Tensor[T] = unsafe { nil }
	w_q       &vtl.Tensor[T] = unsafe { nil }
	w_k       &vtl.Tensor[T] = unsafe { nil }
	w_v       &vtl.Tensor[T] = unsafe { nil }
	w_o       &vtl.Tensor[T] = unsafe { nil }
	num_heads int
	head_dim  int
}

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

pub fn (g &AttentionGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := payload.variable.grad
	d_w_o := la.matmul[T](g.input.transpose([1, 0])!, grad)!
	d_input := la.matmul[T](grad, g.w_o.transpose([1, 0])!)!
	return [d_input, d_w_o, d_w_o, d_w_o, d_w_o]
}

pub fn (g &AttentionGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
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
