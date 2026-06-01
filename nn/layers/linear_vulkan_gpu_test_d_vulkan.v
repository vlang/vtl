module layers

import math
import os
import vtl
import vtl.la

fn test_linear_forward_vulkan_f32_matches_cpu() ! {
	if os.getenv('VTL_USE_VULKAN') != '1' {
		return
	}
	x := vtl.ones[f32]([2, 8])
	w := vtl.ones[f32]([4, 8])
	b := vtl.zeros[f32]([1, 4])
	cpu := la.matmul[f32](x, w.t()!)!.add[f32](b)!
	gpu := linear_forward_vulkan_f32(x, w, b)!
	for i in 0 .. gpu.shape[0] {
		for j in 0 .. gpu.shape[1] {
			diff := math.abs(gpu.get([i, j]) - cpu.get([i, j]))
			assert diff < 0.1, 'linear vulkan mismatch at [${i},${j}]: ${diff}'
		}
	}
}
