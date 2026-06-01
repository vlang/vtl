module layers

// linear_forward_f32_use_vulkan mirrors cuda hooks; true when VTL_USE_VULKAN=1.
@[inline]
pub fn linear_forward_f32_use_vulkan() bool {
	return vulkan_linear_enabled()
}
