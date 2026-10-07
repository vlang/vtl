module internal

import vtl

// embedding_forward looks up each integer index in the weight matrix.
// input: [batch, seq_len] integer indices
// weight: [vocab_size, embedding_dim]
// returns: [batch, seq_len, embedding_dim]
pub fn embedding_forward[T](input &vtl.Tensor[T], weight &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.rank() != 2 || weight.rank() != 2 {
		return error('embedding_forward: expected input [batch, seq_len] and weight [vocab_size, embedding_dim]')
	}
	batch := input.shape[0]
	seq_len := input.shape[1]
	embedding_dim := weight.shape[1]

	mut output := vtl.zeros[T]([batch, seq_len, embedding_dim])
	if input.is_row_major_contiguous() && weight.is_row_major_contiguous()
		&& output.is_row_major_contiguous() {
		indices := input.data.data[..input.size]
		weights := weight.data.data[..weight.size]
		for token in 0 .. batch * seq_len {
			idx := int(vtl.cast[T](indices[token]))
			if idx >= 0 && idx < weight.shape[0] {
				weight_offset := idx * embedding_dim
				output_offset := token * embedding_dim
				for d in 0 .. embedding_dim {
					output.data.data[output_offset + d] = weights[weight_offset + d]
				}
			}
		}
		return output
	}
	for b in 0 .. batch {
		for s in 0 .. seq_len {
			idx := int(input.get([b, s]))
			if idx >= 0 && idx < weight.shape[0] {
				for d in 0 .. embedding_dim {
					output.set([b, s, d], weight.get([idx, d]))
				}
			}
		}
	}
	return output
}

// embedding_backward computes gradient w.r.t. weight.
// Gradients are accumulated into the weight rows corresponding to the input indices.
pub fn embedding_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], weight &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	if input.rank() != 2 || grad_out.rank() != 3 || weight.rank() != 2 {
		return error('embedding_backward: expected input [batch, seq_len], grad_out [batch, seq_len, embedding_dim], and weight [vocab_size, embedding_dim]')
	}
	batch := grad_out.shape[0]
	seq_len := grad_out.shape[1]
	embedding_dim := grad_out.shape[2]
	vocab_size := weight.shape[0]
	if input.shape[0] != batch || input.shape[1] != seq_len || weight.shape[1] != embedding_dim {
		return error('embedding_backward: input, gradient, and weight shapes do not match')
	}

	mut d_weight := vtl.zeros_like[T](weight)
	if input.is_row_major_contiguous() && grad_out.is_row_major_contiguous()
		&& d_weight.is_row_major_contiguous() {
		indices := input.data.data[..input.size]
		gradients := grad_out.data.data[..grad_out.size]
		for token in 0 .. batch * seq_len {
			idx := int(vtl.cast[T](indices[token]))
			if idx >= 0 && idx < vocab_size {
				weight_offset := idx * embedding_dim
				gradient_offset := token * embedding_dim
				for d in 0 .. embedding_dim {
					gradient_index := gradient_offset + d
					d_weight.data.data[weight_offset + d] = vtl.cast[T](f64(d_weight.data.data[weight_offset + d]) + f64(gradients[gradient_index]))
				}
			}
		}
		return [d_weight]
	}

	for b in 0 .. batch {
		for s in 0 .. seq_len {
			idx := int(input.get([b, s]))
			if idx >= 0 && idx < vocab_size {
				for d in 0 .. embedding_dim {
					existing := f64(d_weight.get([idx, d]))
					grad_val := f64(grad_out.get([b, s, d]))
					d_weight.set([idx, d], vtl.cast[T](existing + grad_val))
				}
			}
		}
	}
	return [d_weight]
}
