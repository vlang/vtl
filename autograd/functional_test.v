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
