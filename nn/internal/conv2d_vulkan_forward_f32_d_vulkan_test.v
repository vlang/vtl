module internal

import math
import os
import vtl

fn test_conv2d_forward_vulkan_f32_matches_cpu() ! {
	if os.getenv('VTL_USE_VULKAN') != '1' {
		return
	}
	cfg := Conv2DConfig{
		padding: [1, 1]
		stride:  [1, 1]
	}
	k := [3, 3]
	if !conv2d_vulkan_eligible(k, cfg) {
		return
	}
	input := vtl.ones[f32]([1, 2, 8, 8])
	weight := vtl.ones[f32]([4, 2, 3, 3])
	bias := vtl.zeros[f32]([1, 4])
	cpu := conv2d_forward_cpu_f32(input, weight, bias, k, cfg)!
	gpu := conv2d_forward_vulkan_f32(input, weight, bias, k, cfg)!
	for i in 0 .. cpu.shape[0] {
		for j in 0 .. cpu.shape[1] {
			for h in 0 .. cpu.shape[2] {
				for w in 0 .. cpu.shape[3] {
					diff := math.abs(cpu.get([i, j, h, w]) - gpu.get([i, j, h, w]))
					assert diff < 0.05, 'conv2d vulkan mismatch at [${i},${j},${h},${w}]: ${diff}'
				}
			}
		}
	}
}
