module layers

import vtl.autograd_cuda

pub fn linear_gate_use_cuda_backward() bool {
	return autograd_cuda.cuda_backward_enabled()
}
