module layers

import vtl
import vtl.autograd

fn test_multihead_attention_forward_supports_batched_sequences() ! {
	ctx := autograd.ctx[f64]()
	layer := multihead_attention_layer[f64](ctx, 4, 2)
	input := ctx.variable(vtl.from_array([
		f64(1),
		0.5,
		-0.5,
		0.25,
		0.25,
		-1,
		0.75,
		1,
		-0.5,
		0.25,
		1,
		-0.75,
		1.5,
		-0.25,
		0.5,
		0.125,
		-0.75,
		0.5,
		0.25,
		1.25,
		0.5,
		1.25,
		-0.5,
		0.75,
	], [2, 3, 4])!)

	output := layer.forward(input)!
	assert output.value.shape == [2, 3, 4]
	for i in 0 .. output.value.size() {
		value := output.value.get_nth(i)
		assert value == value, 'attention output contains NaN at flat index ${i}'
	}
}
