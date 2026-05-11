module la

import vtl

// matmul_vulkan fallback: delegates to la.matmul (CPU).
pub fn matmul_vulkan[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[T] {
	result_f64 := matmul[T](a, b)!
	// Convert f64 result back to T
	mut result_array := []T{len: result_f64.size()}
	for i in 0 .. result_f64.size() {
		result_array[i] = vtl.cast[T](result_f64.get_nth(i))
	}
	return vtl.from_1d[T](result_array, vtl.TensorData{}).reshape(result_f64.shape)!
}
