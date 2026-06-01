module layers

import os

pub fn linear_gate_use_vulkan_backward() bool {
	return os.getenv('VTL_USE_VULKAN') == '1'
}
