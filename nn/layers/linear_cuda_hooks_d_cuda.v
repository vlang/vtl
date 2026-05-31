module layers

// linear_forward_f64_use_cuda is true when user opts in via VTL_USE_CUDA=1.
@[inline]
pub fn linear_forward_f64_use_cuda() bool {
	return cuda_linear_enabled()
}
