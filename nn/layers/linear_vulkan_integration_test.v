module layers

import os
import vtl

fn test_linear_forward_f32_cpu_path() ! {
	if os.getenv('VTL_USE_VULKAN') == '1' {
		return
	}
	x := vtl.from_array([f32(1), 2, 3, 4], [2, 2])!
	w := vtl.from_array([f32(0.1), 0.2, 0.3, 0.4], [2, 2])!
	b := vtl.zeros[f32]([1, 2])
	out := linear_forward_f32(x, w, b)!
	assert out.shape == [2, 2]
}

// Sequential f32 forward needs broader f32 autograd gate fixes; use example + VTL_TEST_VULKAN locally.
