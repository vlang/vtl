module internal

import vtl

// embedding_forward looks up each integer index in the weight matrix.
// input: [batch, seq_len] integer indices
// weight: [vocab_size, embedding_dim]
// returns: [batch, seq_len, embedding_dim]
pub fn embedding_forward[T](input &vtl.Tensor[T], weight &vtl.Tensor[T]) !&vtl.Tensor[T] {
	batch := input.shape[0]
	seq_len := input.shape[1]
	embedding_dim := weight.shape[1]

	mut output := vtl.zeros[T]([batch, seq_len, embedding_dim])
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
	batch := grad_out.shape[0]
	seq_len := grad_out.shape[1]
	embedding_dim := grad_out.shape[2]
	vocab_size := weight.shape[0]

	mut d_weight := vtl.zeros_like[T](weight)

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