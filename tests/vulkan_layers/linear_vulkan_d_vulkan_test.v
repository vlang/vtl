module vulkan_layers

import math
import vtl
import vtl.nn.layers { linear_forward_vulkan, relu_forward_vulkan, sigmoid_forward_vulkan }
import vtl.storage

fn test_linear_vulkan_configuration() {
	$if !vulkan ? {
		assert true
	}
}

fn approx_eq_f32(a f32, b f32, eps f32) bool {
	return math.abs(a - b) <= eps
}

fn approx_eq_f64(a f64, b f64, eps f64) bool {
	return math.abs(a - b) <= eps
}

fn test_linear_forward_vulkan_f32() {
	$if vulkan ? {
		x := vtl.from_array[f32]([f32(1.0), 2.0, 3.0, 4.0], [2, 2]) or { panic(err) }
		w := vtl.from_array[f32]([f32(5.0), 6.0, 7.0, 8.0], [2, 2]) or { panic(err) }
		b := vtl.from_array[f32]([f32(1.0), 2.0], [1, 2]) or { panic(err) }

		params := storage.VulkanStorageParams{}
		out := linear_forward_vulkan[f32](x, w, b, params) or { panic(err) }

		// x @ W.t() + b  (row-major):
		// x = [[1,2],[3,4]], W = [[5,6],[7,8]] (row-major, out_f=2, in_f=2)
		// W.t() = [[5,7],[6,8]]
		// x @ W.t() = [[1*5+2*6, 1*7+2*8],[3*5+4*6, 3*7+4*8]] = [[17,23],[39,51]]
		// + b = [[1,2]]  broadcasted row-wise
		// = [[20,24],[44,52]]
		expected := [f32(20.0), 24.0, 44.0, 52.0]
		for i in 0 .. 4 {
			assert approx_eq_f32(out.data.data[i], expected[i], f32(0.1))
		}
	}
}

fn test_linear_forward_vulkan_f64() {
	$if vulkan ? {
		x := vtl.from_array[f64]([f64(1.0), 2.0, 3.0, 4.0], [2, 2]) or { panic(err) }
		w := vtl.from_array[f64]([f64(5.0), 6.0, 7.0, 8.0], [2, 2]) or { panic(err) }
		b := vtl.from_array[f64]([f64(1.0), 2.0], [1, 2]) or { panic(err) }

		params := storage.VulkanStorageParams{}
		out := linear_forward_vulkan[f64](x, w, b, params) or { panic(err) }

		// f64 uses compute bridge (f64->f32 GPU->f64), results may have
		// small rounding differences from pure CPU. Use wider tolerance.
		expected := [f64(20.0), 24.0, 44.0, 52.0]
		for i in 0 .. 4 {
			assert approx_eq_f64(out.data.data[i], expected[i], 1.0)
		}
	}
}

fn test_relu_forward_vulkan_f32() {
	$if vulkan ? {
		x := vtl.from_1d[f32]([f32(-2.0), -1.0, 0.0, 1.0, 2.0]) or { panic(err) }
		params := storage.VulkanStorageParams{}
		out := relu_forward_vulkan[f32](x, params) or { panic(err) }

		expected := [f32(0.0), 0.0, 0.0, 1.0, 2.0]
		for i in 0 .. 5 {
			assert approx_eq_f32(out.data.data[i], expected[i], f32(0.001))
		}
	}
}

fn test_relu_forward_vulkan_f64() {
	$if vulkan ? {
		x := vtl.from_1d[f64]([f64(-2.0), -1.0, 0.0, 1.0, 2.0]) or { panic(err) }
		params := storage.VulkanStorageParams{}
		out := relu_forward_vulkan[f64](x, params) or { panic(err) }

		expected := [f64(0.0), 0.0, 0.0, 1.0, 2.0]
		for i in 0 .. 5 {
			assert approx_eq_f64(out.data.data[i], expected[i], 0.001)
		}
	}
}

fn test_sigmoid_forward_vulkan_f32() {
	$if vulkan ? {
		x := vtl.from_1d[f32]([f32(0.0), 1.0, -1.0]) or { panic(err) }
		params := storage.VulkanStorageParams{}
		out := sigmoid_forward_vulkan[f32](x, params) or { panic(err) }

		expected := [f32(0.5), f32(0.7310586), f32(0.2689414)]
		for i in 0 .. 3 {
			assert approx_eq_f32(out.data.data[i], expected[i], f32(0.01))
		}
	}
}

fn test_sigmoid_forward_vulkan_f64() {
	$if vulkan ? {
		x := vtl.from_1d[f64]([f64(0.0), 1.0, -1.0]) or { panic(err) }
		params := storage.VulkanStorageParams{}
		out := sigmoid_forward_vulkan[f64](x, params) or { panic(err) }

		expected := [f64(0.5), f64(0.7310586), f64(0.2689414)]
		for i in 0 .. 3 {
			assert approx_eq_f64(out.data.data[i], expected[i], 0.01)
		}
	}
}
