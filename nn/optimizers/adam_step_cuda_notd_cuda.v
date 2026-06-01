module optimizers

// adam_step_f64 without CUDA build delegates to CPU (opaque session avoids autograd_cuda import).
pub fn adam_step_f64(grad []f64, mut theta []f64, mut m []f64, mut v []f64, p AdamStepParams,
	session voidptr, slot int) {
	_ = session
	_ = slot
	adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
}
