module vulkan_layers

import vtl
import vtl.nn.layers
import vtl.runtime

fn test_maxpool2d_forward_vulkan_basic() {
	$if vulkan ? {
		mut dev := runtime.new_vulkan_device() or {
			eprintln('no Vulkan device — skipping maxpool2d test')
			return
		}
		defer {
			dev.release() or {}
		}

		// 1×1×4×4 input
		input_data := [
			f32(1),
			2,
			3,
			4,
			f32(5),
			6,
			7,
			8,
			f32(9),
			10,
			11,
			12,
			f32(13),
			14,
			15,
			16,
		]
		input := vtl.from_1d[f32](input_data, vtl.TensorData{})!.reshape([1, 1, 4, 4])!

		kernel_size := [2, 2]!
		stride := [2, 2]!
		padding := [0, 0]!

		out := layers.maxpool2d_forward_vulkan[f32](input, kernel_size, stride, padding,
			dev.device())!

		assert out.shape == [1, 1, 2, 2]

		// expected: max of each 2×2 patch → 6, 8, 14, 16
		assert f32(out.get([0, 0, 0, 0])) == f32(6)
		assert f32(out.get([0, 0, 0, 1])) == f32(8)
		assert f32(out.get([0, 0, 1, 0])) == f32(14)
		assert f32(out.get([0, 0, 1, 1])) == f32(16)
	}
}

fn test_maxpool2d_layer_vulkan() {
	$if vulkan ? {
		mut dev := runtime.new_vulkan_device() or {
			eprintln('no Vulkan device — skipping maxpool2d layer test')
			return
		}
		defer {
			dev.release() or {}
		}

		layer := layers.MaxPool2DLayerVulkan[f32]{
			kernel_size: [2, 2]!
			stride:      [2, 2]!
			padding:     [0, 0]!
			device:      dev.device()
		}

		input_data := [f32(3), 1, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16]
		input := vtl.from_1d[f32](input_data, vtl.TensorData{})!.reshape([1, 1, 4, 4])!

		out := layer.forward(input)!
		assert out.shape == [1, 1, 2, 2]
	}
}
