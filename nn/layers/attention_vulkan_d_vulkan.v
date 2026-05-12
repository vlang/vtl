module layers

import vtl
import vtl.la
import vsl.vulkan
import vtl.storage
import math

// attention_forward_vulkan performs scaled-dot-product attention on Vulkan GPU.
// Q: [batch, num_heads, seq_len, head_dim]
// K: [batch, num_heads, seq_len, head_dim]
// V: [batch, num_heads, seq_len, head_dim]
// Returns: [batch, num_heads, seq_len, head_dim]
pub fn attention_forward_vulkan[T](q &vtl.Tensor[T], k &vtl.Tensor[T], v &vtl.Tensor[T], head_dim int) !&vtl.Tensor[T] {
	if q.shape.len != 4 || k.shape.len != 4 || v.shape.len != 4 {
		return error('attention_forward_vulkan expects 4D tensors: got q=${q.shape}, k=${k.shape}, v=${v.shape}')
	}
	if q.shape != k.shape || q.shape != v.shape {
		return error('attention_forward_vulkan expects matching q/k/v shapes: q=${q.shape}, k=${k.shape}, v=${v.shape}')
	}

	batch := q.shape[0]
	num_heads := q.shape[1]
	seq_len := q.shape[2]
	if q.shape[3] != head_dim {
		return error('attention_forward_vulkan: head_dim mismatch tensor=${q.shape[3]} arg=${head_dim}')
	}

	// Q @ K^T (batched): each [seq_len, head_dim] @ [head_dim, seq_len]
	mut scores := vtl.zeros[T]([batch, num_heads, seq_len, seq_len], q.data)
	for b in 0 .. batch {
		for h in 0 .. num_heads {
			mut q_bh := vtl.zeros[T]([seq_len, head_dim], q.data)
			mut k_bh := vtl.zeros[T]([seq_len, head_dim], k.data)
			for i in 0 .. seq_len {
				for j in 0 .. head_dim {
					q_bh.set([i, j], q.get([b, h, i, j]))
					k_bh.set([i, j], k.get([b, h, i, j]))
				}
			}
			k_t := k_bh.transpose([1, 0])!
			score_bh := la.matmul_vulkan[T](q_bh, k_t)!
			for i in 0 .. seq_len {
				for j in 0 .. seq_len {
					scores.set([b, h, i, j], score_bh.get([i, j]))
				}
			}
		}
	}

	// Scale by 1/sqrt(head_dim)
	scale := vtl.cast[T](1.0 / math.sqrt(f64(head_dim)))
	mut scores_scaled := vtl.zeros_like[T](scores)
	for i in 0 .. scores.size() {
		scores_scaled.set_nth(i, vtl.cast[T](f64(scores.get_nth(i)) * f64(scale)))
	}

	// Softmax on last dimension
	// For now: CPU softmax (GPU softmax needs axis-aware kernel)
	scores_shape := scores_scaled.shape
	mut attn_weights := vtl.zeros_like[T](scores_scaled)

	// Apply softmax row-wise (last dim)
	batch = scores_shape[0]
	num_heads = scores_shape[1]
	seq_len = scores_shape[2]

	for b in 0 .. batch {
		for h in 0 .. num_heads {
			for s in 0 .. seq_len {
				// Extract row
				mut row := []T{len: seq_len}
				for i in 0 .. seq_len {
					row[i] = scores_scaled.get([b, h, s, i])
				}

				// Softmax
				mut max_val := row[0]
				for val in row {
					if f64(val) > f64(max_val) {
						max_val = val
					}
				}
				mut sum := f64(0)
				for i in 0 .. seq_len {
					row[i] = vtl.cast[T](math.exp(f64(row[i]) - f64(max_val)))
					sum += f64(row[i])
				}
				for i in 0 .. seq_len {
					row[i] = vtl.cast[T](f64(row[i]) / sum)
					attn_weights.set([b, h, s, i], row[i])
				}
			}
		}
	}

	// Attn_weights @ V (batched): each [seq_len, seq_len] @ [seq_len, head_dim]
	mut attn_output := vtl.zeros[T]([batch, num_heads, seq_len, head_dim], v.data)
	for b in 0 .. batch {
		for h in 0 .. num_heads {
			mut weights_bh := vtl.zeros[T]([seq_len, seq_len], attn_weights.data)
			mut v_bh := vtl.zeros[T]([seq_len, head_dim], v.data)
			for i in 0 .. seq_len {
				for j in 0 .. seq_len {
					weights_bh.set([i, j], attn_weights.get([b, h, i, j]))
				}
				for j in 0 .. head_dim {
					v_bh.set([i, j], v.get([b, h, i, j]))
				}
			}
			out_bh := la.matmul_vulkan[T](weights_bh, v_bh)!
			for i in 0 .. seq_len {
				for j in 0 .. head_dim {
					attn_output.set([b, h, i, j], out_bh.get([i, j]))
				}
			}
		}
	}
	return attn_output
}

pub fn attention_scores_vulkan[T](q &vtl.Tensor[T], k &vtl.Tensor[T], head_dim int) !&vtl.Tensor[T] {
	if q.shape.len != 4 || k.shape.len != 4 {
		return error('attention_scores_vulkan expects 4D tensors: got q=${q.shape}, k=${k.shape}')
	}
	if q.shape != k.shape {
		return error('attention_scores_vulkan expects matching q/k shapes: q=${q.shape}, k=${k.shape}')
	}
	batch := q.shape[0]
	num_heads := q.shape[1]
	seq_len := q.shape[2]
	if q.shape[3] != head_dim {
		return error('attention_scores_vulkan: head_dim mismatch tensor=${q.shape[3]} arg=${head_dim}')
	}

	mut scores := vtl.zeros[T]([batch, num_heads, seq_len, seq_len], q.data)
	for b in 0 .. batch {
		for h in 0 .. num_heads {
			mut q_bh := vtl.zeros[T]([seq_len, head_dim], q.data)
			mut k_bh := vtl.zeros[T]([seq_len, head_dim], k.data)
			for i in 0 .. seq_len {
				for j in 0 .. head_dim {
					q_bh.set([i, j], q.get([b, h, i, j]))
					k_bh.set([i, j], k.get([b, h, i, j]))
				}
			}
			k_t := k_bh.transpose([1, 0])!
			score_bh := la.matmul_vulkan[T](q_bh, k_t)!
			for i in 0 .. seq_len {
				for j in 0 .. seq_len {
					scores.set([b, h, i, j], score_bh.get([i, j]))
				}
			}
		}
	}
	return scores
}

pub fn attention_apply_values_vulkan[T](weights &vtl.Tensor[T], v &vtl.Tensor[T], head_dim int) !&vtl.Tensor[T] {
	if weights.shape.len != 4 || v.shape.len != 4 {
		return error('attention_apply_values_vulkan expects 4D tensors: got weights=${weights.shape}, v=${v.shape}')
	}
	batch := weights.shape[0]
	num_heads := weights.shape[1]
	seq_len := weights.shape[2]
	if weights.shape[3] != seq_len || v.shape[0] != batch || v.shape[1] != num_heads
		|| v.shape[2] != seq_len || v.shape[3] != head_dim {
		return error('attention_apply_values_vulkan shape mismatch: weights=${weights.shape}, v=${v.shape}, head_dim=${head_dim}')
	}

	mut attn_output := vtl.zeros[T]([batch, num_heads, seq_len, head_dim], v.data)
	for b in 0 .. batch {
		for h in 0 .. num_heads {
			mut weights_bh := vtl.zeros[T]([seq_len, seq_len], weights.data)
			mut v_bh := vtl.zeros[T]([seq_len, head_dim], v.data)
			for i in 0 .. seq_len {
				for j in 0 .. seq_len {
					weights_bh.set([i, j], weights.get([b, h, i, j]))
				}
				for j in 0 .. head_dim {
					v_bh.set([i, j], v.get([b, h, i, j]))
				}
			}
			out_bh := la.matmul_vulkan[T](weights_bh, v_bh)!
			for i in 0 .. seq_len {
				for j in 0 .. head_dim {
					attn_output.set([b, h, i, j], out_bh.get([i, j]))
				}
			}
		}
	}
	return attn_output
}
