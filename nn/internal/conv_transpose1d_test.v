module internal

import vtl
import math

fn test_conv_transpose1d_forward_and_backward() ! {
	input := vtl.from_array([1.0, 2.0], [1, 1, 2])!
	weight := vtl.from_array([1.0, 2.0, 3.0], [1, 1, 3])!
	bias := vtl.from_array([0.5], [1])!
	config := ConvTranspose1DConfig{}
	output := conv_transpose1d_forward[f64](input, weight, bias, config)!
	assert output.shape == [1, 1, 4]
	assert output.to_array() == [1.5, 4.5, 7.5, 6.5]
	grad_output := vtl.ones[f64]([1, 1, 4])
	gradients := conv_transpose1d_backward[f64](grad_output, input, weight, bias, config)!
	assert gradients[0].to_array() == [6.0, 6.0]
	assert gradients[1].to_array() == [3.0, 3.0, 3.0]
	assert gradients[2].to_array() == [4.0]
}

fn test_conv_transpose1d_validates_configuration() {
	input := vtl.ones[f64]([1, 1, 2])
	weight := vtl.ones[f64]([1, 1, 3])
	bias := vtl.zeros[f64]([1])
	if _ := conv_transpose1d_forward[f64](input, weight, bias, ConvTranspose1DConfig{
		stride:         1
		output_padding: 1
	}) {
		assert false, 'output_padding greater than stride and dilation should fail'
	}
	if _ := conv_transpose1d_forward[f64](input, weight, bias, ConvTranspose1DConfig{ groups: 2 }) {
		assert false, 'incompatible groups should fail'
	}
	if _ := conv_transpose1d_forward[f64](input, weight, bias, ConvTranspose1DConfig{
		padding: 2
	}) {
		assert false, 'non-positive calculated output length should fail'
	}
}

fn test_conv_transpose1d_output_padding_and_grouped_channels() ! {
	input := vtl.from_array([1.0], [1, 1, 1])!
	weight := vtl.from_array([2.0, 3.0, 4.0], [1, 1, 3])!
	bias := vtl.from_array([0.5], [1])!
	output := conv_transpose1d_forward[f64](input, weight, bias, ConvTranspose1DConfig{
		stride:         2
		output_padding: 1
	})!
	assert output.shape == [1, 1, 4]
	assert output.to_array() == [2.5, 3.5, 4.5, 0.5]

	grouped_input := vtl.from_array([1.0, 2.0], [1, 2, 1])!
	grouped_weight := vtl.from_array([3.0, 4.0], [2, 1, 1])!
	grouped_bias := vtl.zeros[f64]([2])
	grouped_output := conv_transpose1d_forward[f64](grouped_input, grouped_weight,
		grouped_bias, ConvTranspose1DConfig{ groups: 2 })!
	assert grouped_output.shape == [1, 2, 1]
	assert grouped_output.to_array() == [3.0, 8.0]
}

fn test_conv_transpose1d_gradient_matches_finite_difference() ! {
	input := vtl.from_array([0.3, -0.7], [1, 1, 2])!
	weight := vtl.from_array([0.2, 0.5], [1, 1, 2])!
	bias := vtl.from_array([0.1], [1])!
	grad_output := vtl.from_array([0.4, -0.2, 0.8], [1, 1, 3])!
	analytic := conv_transpose1d_backward[f64](grad_output, input, weight, bias, ConvTranspose1DConfig{})!
	epsilon := 1e-6
	for index in 0 .. weight.size() {
		mut plus := vtl.from_array(weight.to_array(), weight.shape)!
		mut minus := vtl.from_array(weight.to_array(), weight.shape)!
		plus.set_nth(index, plus.get_nth(index) + epsilon)
		minus.set_nth(index, minus.get_nth(index) - epsilon)
		plus_output := conv_transpose1d_forward[f64](input, plus, bias, ConvTranspose1DConfig{})!
		minus_output := conv_transpose1d_forward[f64](input, minus, bias, ConvTranspose1DConfig{})!
		mut plus_loss := 0.0
		mut minus_loss := 0.0
		for i in 0 .. grad_output.size() {
			plus_loss += plus_output.get_nth(i) * grad_output.get_nth(i)
			minus_loss += minus_output.get_nth(i) * grad_output.get_nth(i)
		}
		numeric := (plus_loss - minus_loss) / (2.0 * epsilon)
		assert math.abs(numeric - analytic[1].get_nth(index)) < 1e-7
	}
}
