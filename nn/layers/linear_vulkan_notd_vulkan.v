module layers

@[params]
pub struct VulkanParams {}

// linear_forward_vulkan returns error if Vulkan is not enabled
pub fn linear_forward_vulkan[T](x &Tensor[T], weights &Tensor[T], bias &Tensor[T]) !&Tensor[T] {
    return error(@METHOD + ':' +
        ' it is needed to compile with the flag "-d vulkan" to use the Vulkan Backend')
}

// relu_forward_vulkan returns error if Vulkan is not enabled
pub fn relu_forward_vulkan[T](x &Tensor[T]) !&Tensor[T] {
    return error(@METHOD + ':' +
        ' it is needed to compile with the flag "-d vulkan" to use the Vulkan Backend')
}

// sigmoid_forward_vulkan returns error if Vulkan is not enabled
pub fn sigmoid_forward_vulkan[T](x &Tensor[T]) !&Tensor[T] {
    return error(@METHOD + ':' +
        ' it is needed to compile with the flag "-d vulkan" to use the Vulkan Backend')
}