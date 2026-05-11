module la

import vtl

fn test_matmul_vulkan_basic() {
	// Create 2x3 matrix A
	a_data := [f32(1.0), 2.0, 3.0, 4.0, 5.0, 6.0]
	a := vtl.from_1d[f32](a_data, vtl.TensorData{}).reshape([2, 3]) or {
		eprintln('Failed to create tensor A: ${err}')
		assert false
		return
	}

	// Create 3x2 matrix B
	b_data := [f32(7.0), 8.0, 9.0, 10.0, 11.0, 12.0]
	b := vtl.from_1d[f32](b_data, vtl.TensorData{}).reshape([3, 2]) or {
		eprintln('Failed to create tensor B: ${err}')
		assert false
		return
	}

	// Expected result C = A @ B (2x2)
	// Row 0: [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
	// Row 1: [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]

	// Compute on GPU
	c := matmul_vulkan[f32](a, b) or {
		eprintln('matmul_vulkan failed (no Vulkan device?): ${err}')
		// Skip test gracefully if no Vulkan device
		return
	}

	// Check shape
	assert c.shape[0] == 2
	assert c.shape[1] == 2

	// Check values (with tolerance)
	tol := f32(0.01)
	assert vtl.abs(c.get([0, 0]) - 58.0) < tol
	assert vtl.abs(c.get([0, 1]) - 64.0) < tol
	assert vtl.abs(c.get([1, 0]) - 139.0) < tol
	assert vtl.abs(c.get([1, 1]) - 154.0) < tol

	println('✓ test_matmul_vulkan_basic passed')
}
