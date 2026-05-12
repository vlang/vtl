module runtime

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models

fn test_sequential_backend_setters() {
	mut ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.set_backend(.cpu)
	model.set_backend_strict(true)
	assert ctx.compute_backend == .cpu
	assert ctx.compute_strict
}

fn test_softmax_strict_shape_guard() {
	mut ctx := autograd.ctx[f64]()
	ctx.set_compute_backend(.vulkan)
	ctx.set_compute_strict(true)
	x := vtl.from_2d[f64]([[1.0, 2.0], [3.0, 4.0]]) or { panic(err) }
	input := ctx.variable(x)
	l := layers.softmax_layer[f64](ctx, layers.SoftmaxLayerConfig{})
	if _ := l.forward(input) {
		assert false
	} else {
		assert err.msg().contains('unsupported')
	}
}

fn test_relu_cpu_runtime_dispatch() {
	mut ctx := autograd.ctx[f64]()
	ctx.set_compute_backend(vtl.Backend.cpu)
	x := vtl.from_1d[f64]([-1.0, 0.0, 2.0], vtl.TensorData{}) or { panic(err) }
	input := ctx.variable(x)
	l := layers.relu_layer[f64](ctx, [3])
	out := l.forward(input) or { panic(err) }
	arr := out.value.to_array()
	assert arr[0] == 0.0
	assert arr[1] == 0.0
	assert arr[2] == 2.0
}

fn test_attention_auto_runtime_dispatch_cpu_fallback() {
	mut ctx := autograd.ctx[f64]()
	ctx.set_compute_backend(.auto)
	ctx.set_compute_strict(false)

	x := vtl.from_1d[f64]([
		1.0,
		0.0,
		0.0,
		1.0,
		0.5,
		-0.5,
		-0.5,
		0.5,
	], vtl.TensorData{}) or { panic(err) }
	x3 := x.reshape([1, 2, 4]) or { panic(err) }
	input := ctx.variable(x3)

	l := layers.multihead_attention_layer[f64](ctx, 4, 2)
	out := l.forward(input) or { panic(err) }
	assert out.value.shape == [1, 2, 4]
}

fn test_attention_vulkan_strict_errors_without_runtime_backend() {
	mut ctx := autograd.ctx[f64]()
	ctx.set_compute_backend(.vulkan)
	ctx.set_compute_strict(true)

	x := vtl.from_1d[f64]([
		1.0,
		0.0,
		0.0,
		1.0,
		0.5,
		-0.5,
		-0.5,
		0.5,
	], vtl.TensorData{}) or { panic(err) }
	x3 := x.reshape([1, 2, 4]) or { panic(err) }
	input := ctx.variable(x3)

	l := layers.multihead_attention_layer[f64](ctx, 4, 2)
	if _ := l.forward(input) {
		assert false
	} else {
		assert err.msg().contains('strict mode')
			|| err.msg().contains('not available without -d vulkan')
	}
}
