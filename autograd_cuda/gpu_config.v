module autograd_cuda

import os

// gpu_activations_enabled is true when Phase 2 device-resident forwards are allowed.
// Requires `VTL_USE_CUDA=1`, `VTL_GPU_ACTIVATIONS=1`, and build flag `-d cuda`.

// gpu_activations_enabled exposes this operation as part of the public API.

// gpu_activations_enabled exposes this operation as part of the public API.
@[inline]
pub fn gpu_activations_enabled() bool {
	return os.getenv('VTL_USE_CUDA') == '1' && os.getenv('VTL_GPU_ACTIVATIONS') == '1'
}

// cuda_backward_enabled runs Linear gate GEMMs on GPU (Phase 3).

// cuda_backward_enabled exposes this operation as part of the public API.

// cuda_backward_enabled exposes this operation as part of the public API.
@[inline]
pub fn cuda_backward_enabled() bool {
	return os.getenv('VTL_USE_CUDA') == '1' && os.getenv('VTL_CUDA_BACKWARD') == '1'
}

// cuda_optimizer_enabled runs Adam on GPU with persistent m/v/theta in DeviceSession (#106).

// cuda_optimizer_enabled exposes this operation as part of the public API.

// cuda_optimizer_enabled exposes this operation as part of the public API.
@[inline]
pub fn cuda_optimizer_enabled() bool {
	return os.getenv('VTL_USE_CUDA') == '1' && os.getenv('VTL_CUDA_OPTIMIZER') == '1'
}
