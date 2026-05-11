module main

import vtl
import vtl.storage
import vtl.nn.layers
import math

fn approx_eq_f32(a f32, b f32, eps f32) bool {
	return math.abs(a - b) <= eps
}

fn approx_eq_f64(a f64, b f64, eps f64) bool {
	return math.abs(a - b) <= eps
}

fn test_linear_forward_vulkan_f32() {
	x := vtl.from_array[f32]([f32(1.0), 2.0, 3.0, 4.0], [2, 2]) or { panic(err) }
	w := vtl.from_array[f32]([f32(5.0), 6.0, 7.0, 8.0], [2, 2]) or { panic(err) }
	b := vtl.from_array[f32]([f32(1.0), 2.0], [1, 2]) or { panic(err) }

	params := storage.VulkanStorageParams{}
	out := layers.linear_forward_vulkan[f32](x, w, b, params) or { panic(err) }

	// y = [[1,2],[3,4]] @ [[5,7],[6,8]] + [[1,2]] = [[18,25],[40,55]]
	expected := [f32(18.0), 25.0, 40.0, 55.0]
	for i in 0..4 {
		assert approx_eq_f32(out.data.data[i], expected[i], f32(0.1))
	}
	println('  PASS: linear_forward_vulkan f32 result=${out.data.data}')
}

fn test_linear_forward_vulkan_f64() {
	x := vtl.from_array[f64]([f64(1.0), 2.0, 3.0, 4.0], [2, 2]) or { panic(err) }
	w := vtl.from_array[f64]([f64(5.0), 6.0, 7.0, 8.0], [2, 2]) or { panic(err) }
	b := vtl.from_array[f64]([f64(1.0), 2.0], [1, 2]) or { panic(err) }

	params := storage.VulkanStorageParams{}
	out := layers.linear_forward_vulkan[f64](x, w, b, params) or { panic(err) }

	expected := [f64(18.0), 25.0, 40.0, 55.0]
	for i in 0..4 {
		assert approx_eq_f64(out.data.data[i], expected[i], 0.1)
	}
	println('  PASS: linear_forward_vulkan f64 result=${out.data.data}')
}

fn test_relu_forward_vulkan_f32() {
	x := vtl.from_1d[f32]([f32(-2.0), -1.0, 0.0, 1.0, 2.0]) or { panic(err) }
	params := storage.VulkanStorageParams{}
	out := layers.relu_forward_vulkan[f32](x, params) or { panic(err) }

	expected := [f32(0.0), 0.0, 0.0, 1.0, 2.0]
	for i in 0..5 {
		assert approx_eq_f32(out.data.data[i], expected[i], f32(0.001))
	}
	println('  PASS: relu_forward_vulkan f32')
}

fn test_relu_forward_vulkan_f64() {
	x := vtl.from_1d[f64]([f64(-2.0), -1.0, 0.0, 1.0, 2.0]) or { panic(err) }
	params := storage.VulkanStorageParams{}
	out := layers.relu_forward_vulkan[f64](x, params) or { panic(err) }

	expected := [f64(0.0), 0.0, 0.0, 1.0, 2.0]
	for i in 0..5 {
		assert approx_eq_f64(out.data.data[i], expected[i], 0.001)
	}
	println('  PASS: relu_forward_vulkan f64')
}

fn test_sigmoid_forward_vulkan_f32() {
	x := vtl.from_1d[f32]([f32(0.0), 1.0, -1.0]) or { panic(err) }
	params := storage.VulkanStorageParams{}
	out := layers.sigmoid_forward_vulkan[f32](x, params) or { panic(err) }

	expected := [f32(0.5), f32(0.7310586), f32(0.2689414)]
	for i in 0..3 {
		assert approx_eq_f32(out.data.data[i], expected[i], f32(0.01))
	}
	println('  PASS: sigmoid_forward_vulkan f32')
}

fn test_sigmoid_forward_vulkan_f64() {
	x := vtl.from_1d[f64]([f64(0.0), 1.0, -1.0]) or { panic(err) }
	params := storage.VulkanStorageParams{}
	out := layers.sigmoid_forward_vulkan[f64](x, params) or { panic(err) }

	expected := [f64(0.5), f64(0.7310586), f64(0.2689414)]
	for i in 0..3 {
		assert approx_eq_f64(out.data.data[i], expected[i], 0.01)
	}
	println('  PASS: sigmoid_forward_vulkan f64')
}

fn main() {
	println('--- Vulkan nn.layers forward tests ---')
	test_linear_forward_vulkan_f32()
	test_linear_forward_vulkan_f64()
	test_relu_forward_vulkan_f32()
	test_relu_forward_vulkan_f64()
	test_sigmoid_forward_vulkan_f32()
	test_sigmoid_forward_vulkan_f64()
	println('')
	println('=======================================')
	println(' All Vulkan nn.layers forward tests PASSED ')
	println('=======================================')
}