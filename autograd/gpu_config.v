module autograd

import os

// gpu_activations_enabled is true when Phase 2 device-resident forwards are allowed.
// Requires `VTL_USE_CUDA=1`, `VTL_GPU_ACTIVATIONS=1`, and build flag `-d cuda`.
@[inline]
pub fn gpu_activations_enabled() bool {
	return os.getenv('VTL_USE_CUDA') == '1' && os.getenv('VTL_GPU_ACTIVATIONS') == '1'
}
