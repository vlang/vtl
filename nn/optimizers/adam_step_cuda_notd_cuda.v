module optimizers

import vtl.autograd_cuda

// adam_step_f64 without CUDA build delegates to CPU.
pub fn adam_step_f64(grad []f64, mut theta []f64, mut m []f64, mut v []f64, p AdamStepParams,
	mut session &autograd_cuda.DeviceSession, slot int) {
	_ = session
	_ = slot
	adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
}
