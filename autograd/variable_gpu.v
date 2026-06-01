module autograd

import vtl

// variable_gpu_activation_ptr returns a pointer to the gpu_activation voidptr field.
@[inline]
pub fn variable_gpu_activation_ptr[T](mut v &Variable[T]) &voidptr {
	return &v.gpu_activation
}

// variable_take_gpu_activation_input moves the pending GPU activation out (f64 CUDA builds only).
pub fn variable_take_gpu_activation_input[T](mut v &Variable[T]) voidptr {
	if sizeof(T) != 8 {
		return unsafe { nil }
	}
	if v.gpu_activation == unsafe { nil } {
		return unsafe { nil }
	}
	ptr := v.gpu_activation
	v.gpu_activation = unsafe { nil }
	return ptr
}

// variable_release_gpu_activation releases a stored GPU tensor handle (f64 CUDA).
pub fn variable_release_gpu_activation[T](mut v &Variable[T]) {
	if sizeof(T) != 8 {
		return
	}
	$if cuda ? {
		if v.gpu_activation != unsafe { nil } {
			unsafe { &vtl.CudaTensor[f64](v.gpu_activation) }.release()
			v.gpu_activation = unsafe { nil }
		}
	} $else {
		v.gpu_activation = unsafe { nil }
	}
}

// variable_set_gpu_activation stores a GPU tensor handle (f64 CUDA).
pub fn variable_set_gpu_activation[T](mut v &Variable[T], ptr voidptr) {
	if sizeof(T) != 8 {
		return
	}
	variable_release_gpu_activation(mut v)
	v.gpu_activation = ptr
}
