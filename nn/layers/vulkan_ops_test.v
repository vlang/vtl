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
