module storage

// VulkanStorageParams placeholder when compiling without `-d vulkan`.
// Keeps `Tensor[T].vulkan(storage.VulkanStorageParams)` signatures consistent.
@[params]
pub struct VulkanStorageParams {}
