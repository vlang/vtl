module layers

import os

// linear_gate_use_vulkan_backward exposes this operation as part of the public API.
pub fn linear_gate_use_vulkan_backward() bool {
	return os.getenv('VTL_USE_VULKAN') == '1'
}
