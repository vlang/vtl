module vulkan_layers

import vtl
import vtl.nn.layers
import vtl.runtime
import math

fn test_batchnorm1d_forward_vulkan_basic() {
	$if vulkan ? {
		mut dev := runtime.new_vulkan_device() or {
			eprintln('no Vulkan device — skipping batchnorm1d test')
			return
		}
		defer {
			dev.release() or {}
		}

		// 4 samples, 2 features
		// col0: [1,2,3,4] mean=2.5 var=1.25 → normalised ≈ [-1.342, -0.447, 0.447, 1.342]
		input_data := [f32(1), 2, 2, 4, 3, 6, 4, 8]
		input := vtl.from_1d[f32](input_data, vtl.TensorData{})!.reshape([4, 2])!

		out := layers.batchnorm1d_forward_vulkan[f32](input, f32(1e-5), dev.device())!

		assert out.shape == [4, 2]

		// column 0 normalised values
		assert math.abs(f64(out.get([0, 0])) - (-1.342)) < 0.01
		assert math.abs(f64(out.get([1, 0])) - (-0.447)) < 0.01
		assert math.abs(f64(out.get([2, 0])) - 0.447) < 0.01
		assert math.abs(f64(out.get([3, 0])) - 1.342) < 0.01
	}
}

fn test_batchnorm1d_layer_vulkan() {
	$if vulkan ? {
		mut dev := runtime.new_vulkan_device() or {
			eprintln('no Vulkan device — skipping batchnorm1d layer test')
			return
		}
		defer {
			dev.release() or {}
		}

		layer := layers.BatchNorm1DLayerVulkan[f32]{
			eps:    f32(1e-5)
			device: dev.device()
		}

		input_data := [f32(1), 2, 2, 4, 3, 6, 4, 8]
		input := vtl.from_1d[f32](input_data, vtl.TensorData{})!.reshape([4, 2])!

		out := layer.forward(input)!
		assert out.shape == [4, 2]
	}
}
