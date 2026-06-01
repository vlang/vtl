module autograd

$if cuda ? {
	// take_gpu_activation_input moves a pending GPU activation off this f64 Variable (nn Linear chain).
	pub fn (mut v Variable[f64]) take_gpu_activation_input() voidptr {
		if v.gpu_activation == unsafe { nil } {
			return unsafe { nil }
		}
		ptr := v.gpu_activation
		v.gpu_activation = unsafe { nil }
		return ptr
	}

	pub fn (v Variable[f64]) has_gpu_activation() bool {
		return v.gpu_activation != unsafe { nil }
	}
}
