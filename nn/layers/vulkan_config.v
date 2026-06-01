module layers

import os

// vulkan_linear_enabled is true when the user opts in to Vulkan for NN layers (f32).
// Requires build flag `-d vulkan` and `VTL_USE_VULKAN=1`.
@[inline]
pub fn vulkan_linear_enabled() bool {
	return os.getenv('VTL_USE_VULKAN') == '1'
}

// vulkan_tests_enabled gates optional GPU tests.
@[inline]
pub fn vulkan_tests_enabled() bool {
	return os.getenv('VTL_TEST_VULKAN') == '1'
}
