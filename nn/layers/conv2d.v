module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types

// Conv2D layer: 2D convolution over a 4D input tensor [batch, in_channels, H, W].
// Produces [batch, out_channels, out_H, out_W] output.

// Conv2DConfig defines a public data structure for this module.

// Conv2DConfig defines a public data structure for this module.
@[params]
pub struct Conv2DConfig {
pub:
	padding  []int = [0, 0]
	stride   []int = [1, 1]
	dilation []int = [1, 1]
	groups   int   = 1
}

// Conv2DLayer applies a 2D convolution over a 4D input tensor.
//
// Input:    `[batch, in_channels, H, W]`
// Output:   `[batch, out_channels, out_H, out_W]`
//
// Config options (via `Conv2DConfig`):
//   - `padding`  — zero-padding added to input borders (default: [0,0])
//   - `stride`   — sampling stride in H and W dimensions (default: [1,1])
//   - `dilation` — spacing between kernel elements (default: [1,1])
//   - `groups`   — split input channels into `groups` groups (default: 1)
pub struct Conv2DLayer[T] {
pub:
	in_channels  int
	out_channels int
	kernel_size  []int
	config       Conv2DConfig
	// [C, H, W] from the previous layer when known (for output_shape / flatten).
	input_shape []int
pub mut:
	weight &autograd.Variable[T] = unsafe { nil }
	bias   &autograd.Variable[T] = unsafe { nil }
}

// conv2d_layer creates a Conv2DLayer.
pub fn conv2d_layer[T](ctx &autograd.Context[T], in_ch int, out_ch int, kernel_size []int, config Conv2DConfig, input_shape []int) types.Layer[T] {
	weight_shape := [out_ch, in_ch / config.groups, kernel_size[0], kernel_size[1]]
	weight := internal.kaiming_normal[T](weight_shape)
	bias := vtl.zeros[T]([1, out_ch])
	return types.Layer[T](&Conv2DLayer[T]{
		in_channels:  in_ch
		out_channels: out_ch
		kernel_size:  kernel_size
		config:       config
		input_shape:  input_shape.clone()
		weight:       ctx.variable(weight)
		bias:         ctx.variable(bias)
	})
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &Conv2DLayer[T]) output_shape() []int {
	if layer.input_shape.len >= 3 && layer.input_shape[1] > 0 && layer.input_shape[2] > 0 {
		in_h := layer.input_shape[1]
		in_w := layer.input_shape[2]
		kh := layer.kernel_size[0]
		kw := layer.kernel_size[1]
		ph := layer.config.padding[0]
		pw := layer.config.padding[1]
		sh := layer.config.stride[0]
		sw := layer.config.stride[1]
		out_h := (in_h - kh + 2 * ph) / sh + 1
		out_w := (in_w - kw + 2 * pw) / sw + 1
		return [layer.out_channels, out_h, out_w]
	}
	return [layer.out_channels, -1, -1]
}

// variables exposes this operation as part of the public API.
pub fn (layer &Conv2DLayer[T]) variables() []&autograd.Variable[T] {
	return [layer.weight, layer.bias]
}

// forward exposes this operation as part of the public API.
pub fn (layer &Conv2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	cfg := internal.Conv2DConfig{
		padding:  layer.config.padding
		stride:   layer.config.stride
		dilation: layer.config.dilation
		groups:   layer.config.groups
	}
	output := internal.conv2d_forward[T](input.value, layer.weight.value, layer.bias.value,
		layer.kernel_size, cfg)!
	mut result := input.context.variable(output)

	if input.requires_grad || layer.weight.requires_grad || layer.bias.requires_grad {
		gate := conv2d_gate[T](input.value, layer.weight.value, layer.bias.value,
			layer.kernel_size, layer.config)
		gate.cache(mut result, input, layer.weight, layer.bias)!
	}
	return result
}

// Conv2DGate defines a public data structure for this module.
pub struct Conv2DGate[T] {
	input       &vtl.Tensor[T] = unsafe { nil }
	weight      &vtl.Tensor[T] = unsafe { nil }
	bias        &vtl.Tensor[T] = unsafe { nil }
	kernel_size []int
	config      Conv2DConfig
}

// conv2d_gate exposes this operation as part of the public API.
pub fn conv2d_gate[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T], kernel_size []int, config Conv2DConfig) &Conv2DGate[T] {
	return &Conv2DGate[T]{
		input:       input
		weight:      weight
		bias:        bias
		kernel_size: kernel_size
		config:      config
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &Conv2DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	cfg := internal.Conv2DConfig{
		padding:  g.config.padding
		stride:   g.config.stride
		dilation: g.config.dilation
		groups:   g.config.groups
	}
	return internal.conv2d_backward[T](payload.variable.grad, g.input, g.weight, g.bias,
		g.kernel_size, cfg)
}

// cache exposes this operation as part of the public API.
pub fn (g &Conv2DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	if args.len < 3 {
		return error('Conv2DGate.cache: expected input, weight, bias variables')
	}
	input := args[0]
	weight := args[1]
	bias := args[2]
	match input {
		autograd.Variable[T] {
			match weight {
				autograd.Variable[T] {
					match bias {
						autograd.Variable[T] {
							result.grad = vtl.zeros_like[T](result.value)
							result.requires_grad = true
							autograd.register[T]('Conv2D', g, result, [input, weight, bias])!
						}
						else {
							return error('Conv2DGate: bias must be a Variable')
						}
					}
				}
				else {
					return error('Conv2DGate: weight must be a Variable')
				}
			}
		}
		else {
			return error('Conv2DGate: input must be a Variable')
		}
	}
}
