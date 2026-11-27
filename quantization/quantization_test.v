module quantization

import math
import vtl

fn test_per_tensor_quantize_rounds_to_even_and_saturates() ! {
	params := new_params(1.0, 0, -128, 127)!
	input := vtl.from_1d([-200.0, -1.5, -0.5, 0.5, 1.5, 200.0])!
	got := quantize_per_tensor_f64(input, params)!
	assert got.to_array() == [i8(-128), -2, 0, 0, 2, 127]
	assert got.shape == input.shape

	shifted := new_params(1.0, 1, -128, 127)!
	tie_values := quantize_per_tensor_f64(vtl.from_1d([0.5, -0.5])!, shifted)!
	assert tie_values.to_array() == [i8(2), 0]
}

fn test_per_tensor_quantize_and_dequantize_f32_and_scalar() ! {
	params := new_params(0.25, -3, -128, 127)!
	input := vtl.from_1d[f32]([0.0, 1.0, -1.0])!
	encoded := quantize_per_tensor_f32(input, params)!
	assert encoded.to_array() == [i8(-3), 1, -7]
	decoded := dequantize_per_tensor_f32(encoded, params)!
	assert decoded.shape == input.shape
	assert math.abs(f64(decoded.get_nth(0))) < 1e-6
	assert math.abs(f64(decoded.get_nth(1)) - 1.0) < 1e-6
	assert math.abs(f64(decoded.get_nth(2)) + 1.0) < 1e-6

	scalar := vtl.from_array([2.0], [])!
	scalar_q := quantize_per_tensor_f64(scalar, params)!
	assert scalar_q.shape.len == 0
	assert scalar_q.get_nth(0) == 5
}

fn test_per_channel_quantization_supports_negative_axis_and_strided_input() ! {
	input := vtl.from_array([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], [2, 3])!.transpose([1, 0])!
	params := new_per_channel_params([1.0, 1.0, 1.0], [0, 0, 0], -2, -128,
		127)!
	encoded := quantize_per_channel_f64(input, params)!
	assert encoded.shape == [3, 2]
	assert encoded.to_array() == [i8(1), 5, 4, 3, 2, 6]
	decoded := dequantize_per_channel_f64(encoded, params)!
	assert decoded.to_array() == [1.0, 5.0, 4.0, 3.0, 2.0, 6.0]
}

fn test_per_channel_quantization_uses_channel_parameters() ! {
	input := vtl.from_array([0.1, 0.2, 0.3, 0.2, 0.4, 0.6], [2, 3])!
	params := new_per_channel_params([0.1, 0.2], [0, 0], 0, -128, 127)!
	encoded := quantize_per_channel_f64(input, params)!
	assert encoded.to_array() == [i8(1), 2, 3, 1, 2, 3]
}

fn test_linear_int8_f32_uses_per_output_channel_scales_and_preserves_batch_shape() ! {
	input := vtl.from_array([f32(1), 2, -1, 0.5], [1, 2, 2])!
	input_params := new_params(0.1, 3, -128, 127)!
	weights := vtl.from_array([i8(55), -95, 97, 9], [2, 2])!
	weight_params := new_per_channel_params([0.01, 0.02], [5, -3], -2, -128, 127)!
	bias := vtl.from_array([f32(0.25), -0.5], [1, 2])!

	output := linear_int8_f32(input, weights, input_params, weight_params, bias)!

	assert output.shape == [1, 2, 2]
	expected := [f32(-1.25), 1.98, -0.75, -2.38]
	for i in 0 .. expected.len {
		assert math.abs(f64(output.get_nth(i) - expected[i])) < 1e-5
	}
}

fn test_linear_int8_f32_rejects_mismatched_metadata() ! {
	input := vtl.from_array([f32(1), 2], [1, 2])!
	input_params := new_params(0.1, 0, -128, 127)!
	weights := vtl.from_array([i8(10), 20, 30, 40], [2, 2])!
	bias := vtl.from_1d[f32]([0, 0])!
	good_params := new_per_channel_params([0.1, 0.1], [0, 0], 0, -128, 127)!

	mut axis_failed := false
	linear_int8_f32(input, weights, input_params,
		new_per_channel_params([0.1, 0.1], [0, 0], 1, -128, 127)!, bias) or {
		axis_failed = true
	}
	assert axis_failed

	mut bias_shape_failed := false
	linear_int8_f32(input, weights, input_params, good_params,
		vtl.from_array([f32(0), 0], [2, 1])!) or {
		bias_shape_failed = true
	}
	assert bias_shape_failed

	mut weight_range_failed := false
	linear_int8_f32(input, weights, input_params,
		new_per_channel_params([0.1, 0.1], [0, 0], 0, -128, 39)!, bias) or {
		weight_range_failed = true
	}
	assert weight_range_failed
}

fn test_conv2d_int8_f32_handles_padding_stride_and_affine_zero_points() ! {
	input := vtl.from_array([f32(1), 2, 3, 4, 5, 6], [1, 1, 2, 3])!
	input_params := new_params(0.1, 3, -128, 127)!
	weights := vtl.from_array([i8(55), 105, -55, 20], [2, 1, 1, 2])!
	weight_params := new_per_channel_params([0.01, 0.02], [5, -5], 0, -128, 127)!
	bias := vtl.from_array([f32(0.25), -0.5], [1, 2])!
	config := QuantizedConv2DConfig{
		padding: [0, 1]
		stride:  [1, 2]
	}

	output := conv2d_int8_f32(input, weights, input_params, weight_params, bias, [1, 2],
		config)!

	assert output.shape == [1, 2, 2, 2]
	expected := [f32(1.25), 4.25, 4.25, 8.75, 0, -1, 1.5, -2.5]
	for i in 0 .. expected.len {
		assert math.abs(f64(output.get_nth(i) - expected[i])) < 1e-5
	}
}

fn test_conv2d_int8_f32_supports_groups() ! {
	input := vtl.from_array([f32(1), 2, 3, 4], [1, 2, 1, 2])!
	input_params := new_params(0.1, 0, -128, 127)!
	weights := vtl.from_array([i8(10), 20], [2, 1, 1, 1])!
	weight_params := new_per_channel_params([0.1, 0.2], [0, 0], 0, -128, 127)!
	bias := vtl.from_array([f32(0), 0], [1, 2])!
	config := QuantizedConv2DConfig{
		groups: 2
	}

	output := conv2d_int8_f32(input, weights, input_params, weight_params, bias, [1, 1],
		config)!

	assert output.shape == [1, 2, 1, 2]
	assert output.to_array() == [f32(1), 2, 12, 16]
}

fn test_conv2d_int8_f32_rejects_invalid_groups_and_weight_axis() ! {
	input := vtl.from_array([f32(1)], [1, 1, 1, 1])!
	input_params := new_params(0.1, 0, -128, 127)!
	weights := vtl.from_array([i8(10)], [1, 1, 1, 1])!
	bias := vtl.from_array([f32(0)], [1, 1])!
	weight_params := new_per_channel_params([0.1], [0], 0, -128, 127)!

	mut groups_failed := false
	conv2d_int8_f32(input, weights, input_params, weight_params, bias, [1, 1],
		QuantizedConv2DConfig{
			groups: 2
		}) or {
		groups_failed = true
	}
	assert groups_failed

	mut axis_failed := false
	conv2d_int8_f32(input, weights, input_params,
		new_per_channel_params([0.1], [0], 1, -128, 127)!, bias, [1, 1],
		QuantizedConv2DConfig{}) or {
		axis_failed = true
	}
	assert axis_failed
}

fn test_quantized_layer_wrappers_convert_f32_weights_and_forward() ! {
	input_params := new_params(0.1, 0, -128, 127)!
	linear_input := vtl.from_array([f32(1), 2], [1, 2])!
	linear_weights := vtl.from_array([f32(0.5), -1, 2, 0.24], [2, 2])!
	linear_bias := vtl.from_array([f32(0.25), -0.5], [1, 2])!
	linear_layer := quantize_linear_layer_f32(linear_weights, linear_bias, input_params,
		-128, 127)!
	linear_output := linear_layer.forward(linear_input)!
	expected_linear := [f32(-1.25), 1.98]
	for i in 0 .. expected_linear.len {
		assert math.abs(f64(linear_output.get_nth(i) - expected_linear[i])) < 0.03
	}

	conv_input := vtl.from_array([f32(1), 2], [1, 1, 1, 2])!
	conv_weights := vtl.from_array([f32(0.5)], [1, 1, 1, 1])!
	conv_bias := vtl.from_array([f32(0.25)], [1, 1])!
	conv_layer := quantize_conv2d_layer_f32(conv_weights, conv_bias, input_params, [1, 1],
		QuantizedConv2DConfig{}, -128, 127)!
	conv_output := conv_layer.forward(conv_input)!
	assert conv_output.shape == [1, 1, 1, 2]
	assert conv_output.to_array() == [f32(0.75), 1.25]
}

fn test_minmax_observer_calibrates_and_accumulates() ! {
	mut observer := MinMaxObserver{}
	mut uninitialized_failed := false
	observer.params(-128, 127) or {
		uninitialized_failed = true
	}
	assert uninitialized_failed

	observer.observe_f64(vtl.from_1d([-2.0, 0.5])!)!
	observer.observe_f64(vtl.from_1d([1.0, 3.0])!)!
	assert observer.minimum == -2.0
	assert observer.maximum == 3.0
	params := observer.params(-128, 127)!
	assert params.scale > 0
	assert params.zero_point >= -128 && params.zero_point <= 127

	zero_observer := MinMaxObserver{
		minimum:     0
		maximum:     0
		initialized: true
	}
	zero_params := zero_observer.params(-128, 127)!
	assert zero_params.scale == 1.0
	assert zero_params.zero_point == 0

	mut f32_observer := MinMaxObserver{}
	f32_observer.observe_f32(vtl.from_1d([f32(-0.5), 0.25, 1.0])!)!
	assert f32_observer.minimum == -0.5
	assert f32_observer.maximum == 1.0
}

fn test_affine_params_include_zero_for_one_sided_ranges() ! {
	positive := affine_params(1.0, 2.0, -128, 127)!
	positive_q := quantize_per_tensor_f64(vtl.from_1d([1.0, 2.0])!, positive)!
	assert positive_q.to_array() == [i8(0), 127]
	positive_round_trip := dequantize_per_tensor_f64(positive_q, positive)!
	assert math.abs(positive_round_trip.get_nth(0) - 1.0) < positive.scale
	assert math.abs(positive_round_trip.get_nth(1) - 2.0) < positive.scale

	negative := affine_params(-2.0, -1.0, -128, 127)!
	negative_q := quantize_per_tensor_f64(vtl.from_1d([-2.0, -1.0])!, negative)!
	assert negative_q.to_array() == [i8(-128), 0]
	negative_round_trip := dequantize_per_tensor_f64(negative_q, negative)!
	assert math.abs(negative_round_trip.get_nth(0) + 2.0) < negative.scale
	assert math.abs(negative_round_trip.get_nth(1) + 1.0) < negative.scale
}

fn test_per_channel_observer_calibrates_ranges() ! {
	mut observer := new_per_channel_min_max_observer(0)
	observer.observe_f64(vtl.from_array([-1.0, 1.0, 0.0, 4.0], [2, 2])!)!
	params := observer.params(-128, 127)!
	assert params.scales.len == 2
	assert params.zero_points.len == 2
	encoded := quantize_per_channel_f64(vtl.from_array([-1.0, 1.0, 0.0, 4.0], [
		2,
		2,
	])!, params)!
	decoded := dequantize_per_channel_f64(encoded, params)!
	assert decoded.shape == [2, 2]
	assert math.abs(decoded.get_nth(0) + 1.0) < 0.02
	assert math.abs(decoded.get_nth(3) - 4.0) < 0.02

	f32_input := vtl.from_array([f32(0.1), 0.2, 0.3, 0.2, 0.4, 0.6], [2, 3])!
	f32_params := new_per_channel_params([0.1, 0.2], [0, 0], 0, -128, 127)!
	f32_encoded := quantize_per_channel_f32(f32_input, f32_params)!
	f32_decoded := dequantize_per_channel_f32(f32_encoded, f32_params)!
	assert f32_encoded.to_array() == [i8(1), 2, 3, 1, 2, 3]
	assert math.abs(f64(f32_decoded.get_nth(5)) - 0.6) < 1e-6
}

fn test_quantization_rejects_invalid_parameters_and_non_finite_values() ! {
	mut bad_scale_failed := false
	new_params(0.0, 0, -128, 127) or { bad_scale_failed = true }
	assert bad_scale_failed
	mut bad_zero_failed := false
	new_params(1.0, 128, -128, 127) or { bad_zero_failed = true }
	assert bad_zero_failed
	mut bad_range_failed := false
	new_params(1.0, 0, 3, 2) or { bad_range_failed = true }
	assert bad_range_failed
	mut bad_calibration_failed := false
	affine_params(math.nan(), 1.0, -128, 127) or {
		bad_calibration_failed = true
	}
	assert bad_calibration_failed

	params := new_params(1.0, 0, -128, 127)!
	mut non_finite_failed := false
	quantize_per_tensor_f64(vtl.from_1d([1.0, math.inf(1)])!, params) or {
		non_finite_failed = true
	}
	assert non_finite_failed
	mut observer := MinMaxObserver{}
	observer.observe_f64(vtl.from_1d([1.0, math.nan()])!) or {}
	assert !observer.initialized
	mut channel_observer := new_per_channel_min_max_observer(0)
	channel_observer.observe_f64(vtl.from_array([1.0, math.inf(1), 2.0, 3.0], [2, 2])!) or {}
	assert channel_observer.minimums.len == 0
	mut axis_failed := false
	quantize_per_channel_f64(vtl.from_1d([1.0, 2.0])!, new_per_channel_params([
		1.0,
	], [0], 1, -128, 127)!) or {
		axis_failed = true
	}
	assert axis_failed
}
