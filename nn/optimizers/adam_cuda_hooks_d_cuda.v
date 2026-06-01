module optimizers

import vtl.autograd_cuda

pub fn adam_use_cuda_optimizer() bool {
	return autograd_cuda.cuda_optimizer_enabled()
}
