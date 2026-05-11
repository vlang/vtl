module layers

import storage
import vtl
import vsl.vulkan

// test_softmax_forward_vulkan verifies numerically-stable GPU softmax.
fn test_softmax_forward_vulkan() {
	mut dev := vulkan.new_device() or {
		eprintln('no Vulkan device — skipping softmax test')
		return
	}
	defer {
		dev.release() or {}
	}
	params := storage.new_vulkan_params(dev)

	n := 64
	data := []f32{len: n, init: f32(index) * 0.1}
	x := vtl.from_1d[f32](data, vtl.TensorData{}) or { panic('from_1d: ${err}') }

	result := softmax_forward_vulkan[f32](x, params) or { panic('softmax: ${err}') }

	assert result.size() == n
	mut s := f32(0)
	for i in 0 .. n {
		v := result.get[f32]([i])
		assert v > 0, 'softmax output must be positive'
		s += v
	}
	assert s > f32(0.99) && s < f32(1.01), 'softmax sum=${s} expected ≈ 1'
}

// test_layernorm_forward_vulkan verifies GPU layer normalisation (mean ≈ 0).
fn test_layernorm_forward_vulkan() {
	mut dev := vulkan.new_device() or {
		eprintln('no Vulkan device — skipping layernorm test')
		return
	}
	defer {
		dev.release() or {}
	}
	params := storage.new_vulkan_params(dev)

	n := 64
	data := []f32{len: n, init: f32(index) - f32(n) / 2.0}
	x := vtl.from_1d[f32](data, vtl.TensorData{}) or { panic('from_1d: ${err}') }

	result := layernorm_forward_vulkan[f32](x, f32(1e-5), params) or { panic('layernorm: ${err}') }

	assert result.size() == n
	mut s := f32(0)
	for i in 0 .. n {
		s += result.get[f32]([i])
	}
	mean := s / f32(n)
	assert mean > f32(-0.1) && mean < f32(0.1), 'layernorm mean=${mean} not ≈ 0'
}

// test_reduce_sum_vulkan verifies GPU per-workgroup reduction sum.
fn test_reduce_sum_vulkan() {
	mut dev := vulkan.new_device() or {
		eprintln('no Vulkan device — skipping reduce test')
		return
	}
	defer {
		dev.release() or {}
	}
	params := storage.new_vulkan_params(dev)

	n := 256
	data := []f32{len: n, init: f32(1)} // all ones → sum = n
	x := vtl.from_1d[f32](data, vtl.TensorData{}) or { panic('from_1d: ${err}') }

	partial := reduce_sum_vulkan[f32](x, params) or { panic('reduce: ${err}') }

	mut total := f32(0)
	for v in partial {
		total += v
	}
	assert total > f32(n) - 1.0 && total < f32(n) + 1.0, 'reduce sum=${total} expected ${n}'
}

// test_attention_forward_vulkan verifies GPU-accelerated scaled dot-product attention.
fn test_attention_forward_vulkan() {
	// Skip if no Vulkan device available
	mut dev := vulkan.new_device() or {
		eprintln('no Vulkan device — skipping attention test')
		return
	}
	defer {
		dev.release() or {}
	}

	// Small attention test: batch=1, num_heads=1, seq_len=2, head_dim=2
	batch := 1
	num_heads := 1
	seq_len := 2
	head_dim := 2

	// Q = [[1, 0], [0, 1]] (identity-ish)
	q_data := [f32(1.0), 0.0, 0.0, 1.0]
	mut q_1d := vtl.from_1d[f32](q_data, vtl.TensorData{}) or { panic('from_1d Q: ${err}') }
	q := q_1d.reshape([batch, num_heads, seq_len, head_dim]) or { panic('reshape Q: ${err}') }

	// K = [[1, 0], [0, 1]] (same as Q → attention should be relatively uniform)
	k_data := [f32(1.0), 0.0, 0.0, 1.0]
	mut k_1d := vtl.from_1d[f32](k_data, vtl.TensorData{}) or { panic('from_1d K: ${err}') }
	k := k_1d.reshape([batch, num_heads, seq_len, head_dim]) or { panic('reshape K: ${err}') }

	// V = [[2, 3], [4, 5]] (some values to propagate)
	v_data := [f32(2.0), 3.0, 4.0, 5.0]
	mut v_1d := vtl.from_1d[f32](v_data, vtl.TensorData{}) or { panic('from_1d V: ${err}') }
	v := v_1d.reshape([batch, num_heads, seq_len, head_dim]) or { panic('reshape V: ${err}') }

	// Run attention
	result := attention_forward_vulkan[f32](q, k, v, head_dim) or { panic('attention: ${err}') }

	// Verify shape
	assert result.shape == [batch, num_heads, seq_len, head_dim], 'attention output shape mismatch'

	// Verify output is reasonable (weighted average of V rows)
	// With Q·K^T producing relatively uniform attention, output should be close to mean of V
	out_00 := result.get([0, 0, 0, 0])
	out_01 := result.get([0, 0, 0, 1])
	out_10 := result.get([0, 0, 1, 0])
	out_11 := result.get([0, 0, 1, 1])

	// Values should be between min and max of V
	assert out_00 >= 2.0 && out_00 <= 4.0, 'attention[0,0,0,0]=${out_00} out of range'
	assert out_01 >= 3.0 && out_01 <= 5.0, 'attention[0,0,0,1]=${out_01} out of range'
	assert out_10 >= 2.0 && out_10 <= 4.0, 'attention[0,0,1,0]=${out_10} out of range'
	assert out_11 >= 3.0 && out_11 <= 5.0, 'attention[0,0,1,1]=${out_11} out of range'

	println('✓ test_attention_forward_vulkan passed')
}
