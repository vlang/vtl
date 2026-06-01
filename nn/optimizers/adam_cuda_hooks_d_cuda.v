module optimizers

import vtl.autograd_cuda

// adam_use_cuda_optimizer exposes this operation as part of the public API.
pub fn adam_use_cuda_optimizer() bool {
	return autograd_cuda.cuda_optimizer_enabled()
}
