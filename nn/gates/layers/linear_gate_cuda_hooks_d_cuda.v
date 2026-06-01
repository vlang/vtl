module layers

import vtl.autograd_cuda

// linear_gate_use_cuda_backward exposes this operation as part of the public API.
pub fn linear_gate_use_cuda_backward() bool {
	return autograd_cuda.cuda_backward_enabled()
}
