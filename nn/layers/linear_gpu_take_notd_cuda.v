module layers

// linear_take_gpu_input is a no-op without `-d cuda` (keeps f32 nn builds free of f64 GPU take).
pub fn linear_take_gpu_input(_input voidptr) voidptr {
	return unsafe { nil }
}
