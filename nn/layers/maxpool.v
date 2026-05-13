module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers
import vtl.nn.types
import vtl.runtime
import vsl.compute as vsl_compute

// MaxPool2DLayer is a layer that implements the maxpooling operation.
pub struct MaxPool2DLayer[T] {
	input_shape []int
	kernel      []int
	padding     []int
	stride      []int
}

pub fn maxpool2d_layer[T](ctx &autograd.Context[T], input_shape []int, kernel []int, padding []int, stride []int) types.Layer[T] {
	return types.Layer[T](&MaxPool2DLayer[T]{
		input_shape: input_shape.clone()
		kernel:      kernel.clone()
		padding:     padding.clone()
		stride:      stride.clone()
	})
}

pub fn (layer &MaxPool2DLayer[T]) output_shape() []int {
	c := layer.input_shape[0]
	h := layer.input_shape[1]
	w := layer.input_shape[2]
	kh := layer.kernel[0]
	kw := layer.kernel[1]
	ph := layer.padding[0]
	pw := layer.padding[1]
	sh := layer.stride[0]
	sw := layer.stride[1]

	return [c, (h - kh + 2 * ph) / sh + 1, (w - kw + 2 * pw) / sw + 1]
}

pub fn (layer &MaxPool2DLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &MaxPool2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	backend := input.context.compute_backend
	strict := input.context.compute_strict
	max_indices, cpu_output := internal.maxpool2d[T](input.value, layer.kernel, layer.padding,
		layer.stride)
	mut output := cpu_output
	if backend == .vulkan || backend == .auto {
		mut has_device := true
		mut dev := runtime.new_vulkan_device() or {
			has_device = false
			runtime.VulkanDevice{}
		}
		if has_device {
			defer {
				dev.release() or {}
			}
			k := [layer.kernel[0], layer.kernel[1]]!
			s := [layer.stride[0], layer.stride[1]]!
			p := [layer.padding[0], layer.padding[1]]!
			output = maxpool2d_forward_vulkan[T](input.value, k, s, p, dev.device()) or {
				if strict && backend == .vulkan {
					return err
				}
				cpu_output
			}
		} else if strict && backend == .vulkan {
			available := vsl_compute.available_backends().map(it.str()).join(', ')
			return error('maxpool2d: backend `${backend}` unavailable for this build. available=[${available}]')
		}
	}
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := layers.maxpool2d_gate[T](max_indices, layer.kernel, input.value.shape,
			layer.padding, layer.stride)
		gate.cache(mut result, input, layer.kernel, layer.padding, layer.stride)!
	}

	return result
}
