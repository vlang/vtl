module internal

import math
import os
import vtl

fn test_conv2d_backward_vulkan_f32_d_weight_matches_cpu() ! {
	if os.getenv('VTL_USE_VULKAN') != '1' {
		return
	}
	cfg := Conv2DConfig{
		padding: [0, 0]
		stride:  [1, 1]
	}
	k := [3, 3]
	if !conv2d_vulkan_backward_eligible(k, cfg) {
		return
	}
	input := vtl.ones[f32]([1, 2, 8, 8])
	weight := vtl.ones[f32]([4, 2, 3, 3])
	bias := vtl.zeros[f32]([1, 4])
	grad := vtl.ones[f32]([1, 4, 8, 8])
	cpu := conv2d_backward_cpu_f32(grad, input, weight, bias, k, cfg)!
	gpu := conv2d_backward_vulkan_f32(grad, input, weight, bias, k, cfg)!
	for i in 0 .. cpu[1].shape[0] {
		for j in 0 .. cpu[1].shape[1] {
			for h in 0 .. cpu[1].shape[2] {
				for w in 0 .. cpu[1].shape[3] {
					diff := math.abs(cpu[1].get([i, j, h, w]) - gpu[1].get([i, j, h, w]))
					assert diff < 0.1, 'd_weight vulkan mismatch: ${diff}'
				}
			}
		}
	}
}
