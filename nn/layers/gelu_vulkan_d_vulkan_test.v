module layers

import storage
import vtl
import vtl.runtime
import math

fn test_gelu_forward_vulkan_basic() {
	$if vulkan ? {
		mut dev := runtime.new_vulkan_device() or {
			eprintln('no Vulkan device — skipping gelu test')
			return
		}
		defer {
			dev.release() or {}
		}
		params := storage.new_vulkan_params(dev.device())

		input_data := [f32(0), 1, -1, 2]
		input := vtl.from_1d[f32](input_data, vtl.TensorData{})!

		out := gelu_forward_vulkan[f32](input, params)!

		assert out.shape == [4]

		// gelu(0) ≈ 0
		assert math.abs(f64(out.get([0]))) < 1e-4
		// gelu(1) ≈ 0.8413
		assert math.abs(f64(out.get([1])) - 0.8413) < 1e-3
		// gelu(-1) ≈ -0.1587
		assert math.abs(f64(out.get([2])) - (-0.1587)) < 1e-3
		// gelu(2) ≈ 1.9545
		assert math.abs(f64(out.get([3])) - 1.9545) < 1e-3
	}
}

fn test_gelu_forward_vulkan_shape_preserved() {
	$if vulkan ? {
		mut dev := runtime.new_vulkan_device() or {
			eprintln('no Vulkan device — skipping gelu shape test')
			return
		}
		defer {
			dev.release() or {}
		}
		params := storage.new_vulkan_params(dev.device())

		input_data := [f32(1), 2, 3, 4, 5, 6]
		input := vtl.from_1d[f32](input_data, vtl.TensorData{})!.reshape([2, 3])!

		out := gelu_forward_vulkan[f32](input, params)!

		assert out.shape == [2, 3]
	}
}
