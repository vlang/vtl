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
	// Q @ K^T
	k_t := k.transpose([0, 1, 3, 2])!
	scores := la.matmul_vulkan[T](q, k_t)!

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
	batch := scores_shape[0]
	num_heads := scores_shape[1]
	seq_len := scores_shape[2]

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

	// Attn_weights @ V
	attn_output := la.matmul_vulkan[T](attn_weights, v)!
	return attn_output
}
