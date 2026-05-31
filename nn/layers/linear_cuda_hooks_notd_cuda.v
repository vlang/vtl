module layers

// linear_forward_f64_use_cuda is false without `-d cuda` build.
@[inline]
pub fn linear_forward_f64_use_cuda() bool {
	return false
}
