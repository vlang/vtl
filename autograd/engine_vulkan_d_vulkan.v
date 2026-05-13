module autograd

// Vulkan-accelerated autograd engine helpers.
// Provides variable_vulkan to create Variables whose gradient computation
// will use GPU kernels when backward-pass operations are performed.
// The backward traversal itself is handled by Variable.backprop() in variable.v;
// this file provides the factory and convenience wrappers.
import vtl
import vtl.la
import storage
import vsl.vulkan
import math

// VulkanAutograd holds the Vulkan device context for a training session.
// Create once and pass to all variable_vulkan calls so all ops share the device.
@[heap]
pub struct VulkanAutograd[T] {
pub mut:
	context &Context[T] = unsafe { nil }
	params  storage.VulkanStorageParams
}

// new_vulkan_autograd creates a VulkanAutograd using the best available GPU.
pub fn new_vulkan_autograd[T]() !&VulkanAutograd[T] {
	dev := vulkan.new_device()!
	params := storage.VulkanStorageParams{
		device: dev
	}
	return &VulkanAutograd[T]{
		context: ctx[T]()
		params:  params
	}
}

// variable creates a Variable tracked in this autograd context.
pub fn (va &VulkanAutograd[T]) variable[T](value &vtl.Tensor[T], requires_grad bool) &Variable[T] {
	return variable[T](va.context, value, requires_grad: requires_grad)
}

// relu_backward performs relu and registers ReLUGateVulkan for GPU backward.
pub fn (va &VulkanAutograd[T]) relu[T](a &Variable[T]) !&Variable[T] {
	result_value := a.value.map(fn [T](val T, _ []int) T {
		zero := vtl.cast[T](0)
		return if val > zero { val } else { zero }
	})
	mut result := variable[T](va.context, result_value, requires_grad: a.is_grad_needed())
	if a.is_grad_needed() {
		gate := relu_gate_vulkan[T](a, va.params)
		gate.cache[T](mut result, a)!
	}
	return result
}

// sigmoid_backward performs sigmoid and registers SigmoidGateVulkan for GPU backward.
pub fn (va &VulkanAutograd[T]) sigmoid[T](a &Variable[T]) !&Variable[T] {
	result_value := a.value.map(fn [T](val T, _ []int) T {
		one := vtl.cast[T](1)
		// sigmoid(x) = 1 / (1 + exp(-x))
		return one / (one + T(math.exp(f64(-val))))
	})
	mut result := variable[T](va.context, result_value, requires_grad: a.is_grad_needed())
	if a.is_grad_needed() {
		gate := sigmoid_gate_vulkan[T](result_value, va.params)
		gate.cache[T](mut result, a)!
	}
	return result
}

// tanh performs tanh and registers TanhGateVulkan for GPU backward.
pub fn (va &VulkanAutograd[T]) tanh[T](a &Variable[T]) !&Variable[T] {
	result_value := a.value.map(fn [T](val T, _ []int) T {
		return T(math.tanh(f64(val)))
	})
	mut result := variable[T](va.context, result_value, requires_grad: a.is_grad_needed())
	if a.is_grad_needed() {
		gate := tanh_gate_vulkan[T](result_value, va.params)
		gate.cache[T](mut result, a)!
	}
	return result
}

// matmul performs matrix multiplication and registers MatMulGateVulkan for GPU backward.
pub fn (va &VulkanAutograd[T]) matmul[T](a &Variable[T], b &Variable[T]) !&Variable[T] {
	result_value := la.matmul[T](a.value, b.value)!
	mut result := variable[T](va.context, result_value,
		requires_grad: a.is_grad_needed() || b.is_grad_needed()
	)
	if a.is_grad_needed() || b.is_grad_needed() {
		gate := matmul_gate_vulkan[T](a, b, va.params)
		gate.cache[T](mut result, a, b)!
	}
	return result
}
