module layers

// linear_forward_f64_use_cuda is true when user opts in via VTL_USE_CUDA=1.

// linear_forward_f64_use_cuda exposes this operation as part of the public API.

// linear_forward_f64_use_cuda exposes this operation as part of the public API.
@[inline]
pub fn linear_forward_f64_use_cuda() bool {
	return cuda_linear_enabled()
}
