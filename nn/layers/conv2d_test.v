module layers

import vtl
import vtl.autograd

fn test_conv2d_layer_output_shape_accounts_for_dilation() {
	ctx := autograd.ctx[f64]()
	layer := conv2d_layer[f64](ctx, 2, 4, [3, 3], Conv2DConfig{
		dilation: [2, 2]
	}, [2, 9, 11])

	assert layer.output_shape() == [4, 5, 7]
}

fn test_conv2d_layer_forward_shape_accounts_for_dilation() ! {
	ctx := autograd.ctx[f64]()
	layer := conv2d_layer[f64](ctx, 1, 2, [3, 3], Conv2DConfig{
		dilation: [2, 2]
	}, [1, 7, 7])
	input := ctx.variable(vtl.zeros[f64]([1, 1, 7, 7]))

	output := layer.forward(input)!

	assert output.value.shape == [1, 2, 3, 3]
}
