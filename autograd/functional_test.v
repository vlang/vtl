module autograd

import vtl

fn test_variable_vmap_preserves_gradients() ! {
	c := ctx[f64]()
	input := c.variable(vtl.from_array([1.0, 2.0, 3.0], [3])!, requires_grad: true)
	result := vmap[f64](fn (sample &Variable[f64]) !&Variable[f64] {
		return sample.add(sample)
	}, input, 0, 0)!
	loss := result.sum()!
	loss.backprop()!
	assert result.value.to_array() == [2.0, 4.0, 6.0]
	assert input.grad.to_array() == [2.0, 2.0, 2.0]
}

fn test_variable_vmap_accepts_output_axis_at_rank_end() ! {
	c := ctx[f64]()
	input := c.variable(vtl.from_array([2.0, 3.0], [2, 1])!, requires_grad: true)
	result := vmap[f64](fn (sample &Variable[f64]) !&Variable[f64] {
		return sample.reshape([1, 1])
	}, input, 0, 2)!
	assert result.value.shape == [1, 1, 2]
	mut loss := result.sum()!
	loss.backprop()!
	assert input.grad.to_array() == [1.0, 1.0]
}
