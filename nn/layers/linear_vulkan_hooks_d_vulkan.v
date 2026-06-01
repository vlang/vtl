module layers

// linear_forward_f32_use_vulkan mirrors cuda hooks; true when VTL_USE_VULKAN=1.

// linear_forward_f32_use_vulkan exposes this operation as part of the public API.

// linear_forward_f32_use_vulkan exposes this operation as part of the public API.
@[inline]
pub fn linear_forward_f32_use_vulkan() bool {
	return vulkan_linear_enabled()
}
