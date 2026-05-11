module layers

// Stub for non-Vulkan builds — embedding_forward_vulkan falls through to CPU.
import vtl

pub fn embedding_forward_vulkan[T](input &vtl.Tensor[T], weight &vtl.Tensor[T]) !&vtl.Tensor[T] {
	batch := input.shape[0]
	seq_len := input.shape[1]
	embed_dim := weight.shape[1]
	mut output := vtl.zeros[T]([batch, seq_len, embed_dim])
	for b in 0 .. batch {
		for s in 0 .. seq_len {
			idx := int(input.get([b, s]))
			if idx >= 0 && idx < weight.shape[0] {
				for d in 0 .. embed_dim {
					output.set([b, s, d], weight.get([idx, d]))
				}
			}
		}
	}
	return output
}
