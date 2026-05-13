module layers

import vtl
import vtl.runtime

fn test_avgpool2d_forward_vulkan() {
	$if vulkan ? {
		mut dev := runtime.new_vulkan_device() or {
			println('No Vulkan device available, skipping test')
			return
		}
		defer {
			dev.release() or { panic(err) }
		}

		// Test: 1×1×4×4 input, 2×2 kernel, stride 2, no padding
		// Output should be 1×1×2×2
		input_data := [
			// Channel 0:
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

		input := vtl.from_1d[f32](input_data, vtl.TensorData{})!
		input_reshaped := input.reshape([1, 1, 4, 4])!

		kernel_size := [2, 2]!
		stride := [2, 2]!
		padding := [0, 0]!

		output := avgpool2d_forward_vulkan[f32](input_reshaped, kernel_size, stride, padding,
			dev.device())!

		assert output.shape == [1, 1, 2, 2]

		// Expected:
		// Top-left: (1+2+5+6)/4 = 14/4 = 3.5
		// Top-right: (3+4+7+8)/4 = 22/4 = 5.5
		// Bottom-left: (9+10+13+14)/4 = 46/4 = 11.5
		// Bottom-right: (11+12+15+16)/4 = 54/4 = 13.5
		expected := [f32(3.5), 5.5, 11.5, 13.5]

		for i in 0 .. 4 {
			b := i / 2
			c := i % 2
			val := output.get([0, 0, b, c])
			diff := if val > expected[i] { val - expected[i] } else { expected[i] - val }
			assert diff < 0.01, 'output[${i}] = ${val}, expected ${expected[i]}'
		}

		println('✓ test_avgpool2d_forward_vulkan passed')
	}
}

fn test_global_avgpool2d_forward_vulkan() {
	$if vulkan ? {
		mut dev := runtime.new_vulkan_device() or {
			println('No Vulkan device available, skipping test')
			return
		}
		defer {
			dev.release() or { panic(err) }
		}

		// Test: 2×2×2×2 input (batch=2, channels=2, height=2, width=2)
		input_data := [
			// batch 0, channel 0: [1,2,3,4] → avg = 2.5
			f32(1),
			2,
			3,
			4,
			// batch 0, channel 1: [5,6,7,8] → avg = 6.5
			f32(5),
			6,
			7,
			8,
			// batch 1, channel 0: [9,10,11,12] → avg = 10.5
			f32(9),
			10,
			11,
			12,
			// batch 1, channel 1: [13,14,15,16] → avg = 14.5
			f32(13),
			14,
			15,
			16,
		]

		input := vtl.from_1d[f32](input_data, vtl.TensorData{})!
		input_reshaped := input.reshape([2, 2, 2, 2])!

		output := global_avgpool2d_forward_vulkan[f32](input_reshaped, dev.device())!

		assert output.shape == [2, 2, 1, 1]

		expected := [f32(2.5), 6.5, 10.5, 14.5]
		for i in 0 .. 4 {
			b := i / 2
			c := i % 2
			val := output.get([b, c, 0, 0])
			diff := if val > expected[i] { val - expected[i] } else { expected[i] - val }
			assert diff < 0.01, 'output[${i}] = ${val}, expected ${expected[i]}'
		}

		println('✓ test_global_avgpool2d_forward_vulkan passed')
	}
}
