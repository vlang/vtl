module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types

// Conv2D layer: 2D convolution over a 4D input tensor [batch, in_channels, H, W].
// Produces [batch, out_channels, out_H, out_W] output.
@[params]
pub struct Conv2DConfig {
	padding     []int = [0, 0]
	stride      []int = [1, 1]
	dilation    []int = [1, 1]
	groups      int = 1
}

pub struct Conv2DLayer[T] {
	in_channels  int
	out_channels int
	kernel_size  []int
	config       Conv2DConfig
pub mut:
	weight       &autograd.Variable[T] = unsafe { nil }
	bias         &autograd.Variable[T] = unsafe { nil }
}

pub fn conv2d_layer[T](ctx &autograd.Context[T], in_ch int, out_ch int, kernel_size []int, config Conv2DConfig) types.Layer[T] {
	weight_shape := [out_ch, in_ch / config.groups, kernel_size[0], kernel_size[1]]
	weight := internal.kaiming_normal[T](weight_shape)
	bias := vtl.zeros[T]([1, out_ch])
	return types.Layer[T](&Conv2DLayer[T]{
		in_channels:  in_ch
		out_channels: out_ch
		kernel_size:  kernel_size
		config:       config
		weight:       ctx.variable(weight)
		bias:         ctx.variable(bias)
	})
}

pub fn (layer &Conv2DLayer[T]) output_shape() []int {
	shapes := [layer.out_channels, -1, -1]
	return shapes
}

pub fn (layer &Conv2DLayer[T]) variables() []&autograd.Variable[T] {
	return [layer.weight, layer.bias]
}

pub fn (layer &Conv2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	cfg := internal.Conv2DConfig{
		padding:  layer.config.padding
		stride:   layer.config.stride
		dilation: layer.config.dilation
		groups:   layer.config.groups
	}
	output := internal.conv2d_forward[T](
		input.value,
		layer.weight.value,
		layer.bias.value,
		layer.kernel_size,
		cfg
	)!
	mut result := input.context.variable(output)

	if input.requires_grad || layer.weight.requires_grad || layer.bias.requires_grad {
		gate := conv2d_gate[T](input.value, layer.weight.value, layer.bias.value, layer.kernel_size, layer.config)
		gate.cache(mut result, input)!
	}
	return result
}

pub struct Conv2DGate[T] {
	input       &vtl.Tensor[T] = unsafe { nil }
	weight      &vtl.Tensor[T] = unsafe { nil }
	bias        &vtl.Tensor[T] = unsafe { nil }
	kernel_size []int
	config      Conv2DConfig
}

pub fn conv2d_gate[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T], kernel_size []int, config Conv2DConfig) &Conv2DGate[T] {
	return &Conv2DGate[T]{
		input: input, weight: weight, bias: bias, kernel_size: kernel_size, config: config
	}
}

pub fn (g &Conv2DGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	cfg := internal.Conv2DConfig{
		padding:  g.config.padding
		stride:   g.config.stride
		dilation: g.config.dilation
		groups:   g.config.groups
	}
	return internal.conv2d_backward[T](
		payload.variable.grad,
		g.input, g.weight, g.bias, g.kernel_size, cfg
	)
}

pub fn (g &Conv2DGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Conv2D', g, result, [a])!
		}
		else {}
	}
}