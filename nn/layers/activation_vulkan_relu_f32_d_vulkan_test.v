module layers

import math
import os
import vtl
import vtl.nn.internal

fn test_relu_forward_vulkan_f32_matches_cpu() ! {
	if os.getenv('VTL_USE_VULKAN') != '1' {
		return
	}
	x := vtl.from_array([f32(2), -1, 0.5, -3], [2, 2])!
	cpu := internal.relu[f32](x)
	gpu := relu_forward_vulkan_f32(x)!
	for i in 0 .. cpu.shape[0] {
		for j in 0 .. cpu.shape[1] {
			diff := math.abs(cpu.get([i, j]) - gpu.get([i, j]))
			assert diff < 1e-3, 'relu vulkan mismatch at [${i},${j}]: ${diff}'
		}
	}
}
