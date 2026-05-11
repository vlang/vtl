module layers

import vtl
import vsl.vulkan

// embedding_forward_vulkan gathers rows from the weight matrix using GPU.
// input: [batch, seq_len] integer indices (stored as T, cast to u32)
// weight: [vocab_size, embed_dim] float32 embedding table
// output: [batch, seq_len, embed_dim]
pub fn embedding_forward_vulkan[T](input &vtl.Tensor[T], weight &vtl.Tensor[T]) !&vtl.Tensor[T] {
	$if T !is f32 {
		// f64 not natively supported by GPU kernel — fall back to CPU
		return embedding_forward_cpu[T](input, weight)
	}
	batch := input.shape[0]
	seq_len := input.shape[1]
	vocab_size := u32(weight.shape[0])
	embed_dim := u32(weight.shape[1])
	num_indices := u32(batch * seq_len)

	mut dev := vulkan.new_device() or {
		return embedding_forward_cpu[T](input, weight)
	}
	defer { dev.release() or {} }

	// Upload flat indices as u32
	mut idx_bytes := []u8{len: int(num_indices * 4)}
	for i in 0 .. int(num_indices) {
		b := i / seq_len
		s := i % seq_len
		val := u32(input.get([b, s]))
		unsafe { *(&u32(&idx_bytes[i * 4])) = val }
	}
	mut idx_buf := dev.buffer(vulkan.DeviceSize(u64(num_indices) * 4))!
	defer { idx_buf.release() }
	idx_buf.load(idx_bytes)!

	// Upload embedding table as f32
	table_size := int(vocab_size * embed_dim)
	mut table_bytes := []u8{len: table_size * 4}
	for i in 0 .. table_size {
		r := i / int(embed_dim)
		c := i % int(embed_dim)
		val := f32(weight.get([r, c]))
		unsafe { *(&f32(&table_bytes[i * 4])) = val }
	}
	mut table_buf := dev.buffer(vulkan.DeviceSize(u64(table_size) * 4))!
	defer { table_buf.release() }
	table_buf.load(table_bytes)!

	// Output buffer: [num_indices, embed_dim]
	mut out_buf := dev.buffer(vulkan.DeviceSize(u64(num_indices * embed_dim) * 4))!
	defer { out_buf.release() }

	vulkan.embedding_gather(dev, out_buf, idx_buf, table_buf, num_indices, vocab_size, embed_dim)!

	mut raw := []u8{len: int(num_indices * embed_dim * 4)}
	out_buf.store(mut raw)!

	mut result_data := []T{len: int(num_indices * embed_dim)}
	for i in 0 .. result_data.len {
		unsafe { result_data[i] = T(*(&f32(&raw[i * 4]))) }
	}

	mut t := vtl.from_1d[T](result_data, vtl.TensorData{})!
	return t.reshape([batch, seq_len, int(embed_dim)])!
}

// embedding_forward_cpu is the pure-V fallback used for non-f32 types.
fn embedding_forward_cpu[T](input &vtl.Tensor[T], weight &vtl.Tensor[T]) !&vtl.Tensor[T] {
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
