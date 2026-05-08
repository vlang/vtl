module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types

pub struct AveragePool2DLayer[T] {
	kernel []int
	padding []int
	stride []int
	input_shape []int
}

pub fn avgpool2d_layer[T](ctx &autograd.Context[T], input_shape []int, kernel []int, padding []int, stride []int) types.Layer[T] {
	return types.Layer[T](&AveragePool2DLayer[T]{
		kernel: kernel.clone(), padding: padding.clone(), stride: stride.clone(), input_shape: input_shape.clone()
	})
}

pub fn (layer &AveragePool2DLayer[T]) output_shape() []int {
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

pub fn (layer &AveragePool2DLayer[T]) variables() []&autograd.Variable[T] { return [] }
pub fn (layer &AveragePool2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.avgpool2d_forward[T](input.value, layer.kernel, layer.padding, layer.stride)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := avgpool2d_gate[T](input.value, layer.kernel, layer.padding, layer.stride)
		gate.cache(mut result, input)!
	}
	return result
}

pub struct AvgPool2DGate[T] {
	input &vtl.Tensor[T] = unsafe { nil }
	kernel []int
	padding []int
	stride []int
}

pub fn avgpool2d_gate[T](input &vtl.Tensor[T], kernel []int, padding []int, stride []int) &AvgPool2DGate[T] {
	return &AvgPool2DGate[T]{input: input, kernel: kernel, padding: padding, stride: stride}
}

pub fn (g &AvgPool2DGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return internal.avgpool2d_backward[T](payload.variable.grad, g.kernel, g.padding, g.stride)
}

pub fn (g &AvgPool2DGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	match args[0] {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('AvgPool2D', g, result, [args[0]])!
		}
		else {}
	}
}

pub struct GlobalAvgPool2DLayer[T] {}

pub fn global_avgpool2d_layer[T](ctx &autograd.Context[T]) types.Layer[T] {
	return types.Layer[T](&GlobalAvgPool2DLayer[T]{})
}

pub fn (layer &GlobalAvgPool2DLayer[T]) output_shape() []int { return []int{-1, -1} }
pub fn (layer &GlobalAvgPool2DLayer[T]) variables() []&autograd.Variable[T] { return [] }
pub fn (layer &GlobalAvgPool2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.global_avgpool2d_forward[T](input.value)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := global_avgpool2d_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}

pub struct GlobalAvgPool2DGate[T] {
	input &vtl.Tensor[T] = unsafe { nil }
}

pub fn global_avgpool2d_gate[T](input &vtl.Tensor[T]) &GlobalAvgPool2DGate[T] {
	return &GlobalAvgPool2DGate[T]{input: input}
}

pub fn (g &GlobalAvgPool2DGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return [internal.global_avgpool2d_backward[T](payload.variable.grad, g.input)]
}

pub fn (g &GlobalAvgPool2DGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	match args[0] {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('GlobalAvgPool2D', g, result, [args[0]])!
		}
		else {}
	}
}
