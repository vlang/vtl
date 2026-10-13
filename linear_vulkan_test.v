// vtest build: false // requires a Vulkan SDK and device
module main

import math
import vtl
import vtl.nn.layers

// Test Vulkan linear layer forward pass
fn test_linear_forward_vulkan() {
	// Create test data: x [2, 3], W [4, 3], b [4]
	mut x := vtl.zeros[f32]([2, 3])
	x.set([0, 0], 1.0)
	x.set([0, 1], 2.0)
	x.set([0, 2], 3.0)
	x.set([1, 0], 4.0)
	x.set([1, 1], 5.0)
	x.set([1, 2], 6.0)

	mut w := vtl.zeros[f32]([4, 3])
	w.set([0, 0], 0.1)
	w.set([0, 1], 0.2)
	w.set([0, 2], 0.3)
	w.set([1, 0], 0.4)
	w.set([1, 1], 0.5)
	w.set([1, 2], 0.6)
	w.set([2, 0], 0.7)
	w.set([2, 1], 0.8)
	w.set([2, 2], 0.9)
	w.set([3, 0], 1.0)
	w.set([3, 1], 1.1)
	w.set([3, 2], 1.2)

	mut b := vtl.zeros[f32]([4])
	b.set([0], 0.1)
	b.set([1], 0.2)
	b.set([2], 0.3)
	b.set([3], 0.4)

	// Test Vulkan path
	result := layers.linear_forward_vulkan(x, w, b)!
	println('Linear forward result: ${result}')

	// Expected: y = x * W^T + b with W as [out_features, in_features]
	// (PyTorch nn.Linear semantics):
	// row 0: [1*.1+2*.2+3*.3, 1*.4+2*.5+3*.6, 1*.7+2*.8+3*.9, 1*1+2*1.1+3*1.2] + b
	//      = [1.4, 3.2, 5.0, 6.8] + [0.1, 0.2, 0.3, 0.4] = [1.5, 3.4, 5.3, 7.2]
	// row 1: [4*.1+5*.2+6*.3, 4*.4+5*.5+6*.6, 4*.7+5*.8+6*.9, 4*1+5*1.1+6*1.2] + b
	//      = [3.2, 7.7, 12.2, 16.7] + b = [3.3, 7.9, 12.5, 17.1]

	expected := vtl.zeros[f32]([2, 4])
	expected.set([0, 0], 1.5)
	expected.set([0, 1], 3.4)
	expected.set([0, 2], 5.3)
	expected.set([0, 3], 7.2)
	expected.set([1, 0], 3.3)
	expected.set([1, 1], 7.9)
	expected.set([1, 2], 12.5)
	expected.set([1, 3], 17.1)

	// Check if close enough (allowing for small floating point differences)
	res_arr := result.to_array()
	exp_arr := expected.to_array()
	assert res_arr.len == exp_arr.len
	for i in 0 .. res_arr.len {
		diff := math.abs(f64(res_arr[i] - exp_arr[i]))
		assert diff < 1e-5, 'Mismatch at index ${i}: got ${res_arr[i]}, expected ${exp_arr[i]}'
	}

	println('✓ Linear forward test passed')

}

// Test ReLU activation
fn test_relu_forward_vulkan() {
	mut x := vtl.zeros[f32]([4])
	x.set([0], -2.0)
	x.set([1], -0.5)
	x.set([2], 1.0)
	x.set([3], 2.0)
	result := layers.relu_forward_vulkan(x)!
	println('ReLU result: ${result}')

	expected := vtl.zeros[f32]([4])
	expected.set([0], 0.0)
	expected.set([1], 0.0)
	expected.set([2], 1.0)
	expected.set([3], 2.0)

	res_arr := result.to_array()
	exp_arr := expected.to_array()
	assert res_arr.len == exp_arr.len
	for i in 0 .. res_arr.len {
		diff := math.abs(f64(res_arr[i] - exp_arr[i]))
		assert diff < 1e-5
	}

	println('✓ ReLU forward test passed')

}

// Test Sigmoid activation
fn test_sigmoid_forward_vulkan() {
	mut x := vtl.zeros[f32]([2])
	x.set([0], 0.0)
	x.set([1], 1.0)
	result := layers.sigmoid_forward_vulkan(x)!
	println('Sigmoid result: ${result}')

	// sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731
	expected := vtl.zeros[f32]([2])
	expected.set([0], 0.5)
	expected.set([1], 0.731)

	res_arr := result.to_array()
	exp_arr := expected.to_array()
	assert res_arr.len == exp_arr.len
	for i in 0 .. res_arr.len {
		diff := math.abs(f64(res_arr[i] - exp_arr[i]))
		assert diff < 1e-2 // Looser tolerance for sigmoid
	}

	println('✓ Sigmoid forward test passed')

}

fn main() {
	println('Testing Vulkan NN layers...')
	test_linear_forward_vulkan()
	test_relu_forward_vulkan()
	test_sigmoid_forward_vulkan()
	println('All tests passed!')
}
