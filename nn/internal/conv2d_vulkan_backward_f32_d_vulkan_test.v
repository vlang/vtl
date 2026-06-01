module internal

import math
import os
import vtl

fn test_conv2d_backward_vulkan_f32_d_weight_matches_cpu() ! {
	if os.getenv('VTL_USE_VULKAN') != '1' {
		return
	}
	k := [3, 3]
	input := vtl.ones[f32]([1, 2, 8, 8])
	weight := vtl.ones[f32]([4, 2, 3, 3])
	bias := vtl.zeros[f32]([1, 4])

	for cfg in [
		Conv2DConfig{
			padding: [0, 0]
			stride:  [1, 1]
		},
		Conv2DConfig{
			padding: [1, 1]
			stride:  [1, 1]
		},
	] {
		if !conv2d_vulkan_backward_eligible(k, cfg) {
			continue
		}
		out_h := (input.shape[2] + 2 * cfg.padding[0] - k[0]) / cfg.stride[0] + 1
		out_w := (input.shape[3] + 2 * cfg.padding[1] - k[1]) / cfg.stride[1] + 1
		grad := vtl.ones[f32]([1, 4, out_h, out_w])
		cpu := conv2d_backward_cpu_f32(grad, input, weight, bias, k, cfg)!
		gpu := conv2d_backward_vulkan_f32(grad, input, weight, bias, k, cfg)!
		for i in 0 .. cpu[1].shape[0] {
			for j in 0 .. cpu[1].shape[1] {
				for h in 0 .. cpu[1].shape[2] {
					for w in 0 .. cpu[1].shape[3] {
						diff := math.abs(cpu[1].get([i, j, h, w]) - gpu[1].get([i, j, h, w]))
						assert diff < 0.1, 'd_weight vulkan mismatch pad=${cfg.padding}: ${diff}'
					}
				}
			}
		}
	}
}
