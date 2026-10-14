module quantization

import math
import vtl

// QuantizationParams describes affine signed INT8 quantization.
pub struct QuantizationParams {
pub:
	scale      f64
	zero_point int
	qmin       i8 = -128
	qmax       i8 = 127
}

// PerChannelParams stores one affine scale and zero point for each channel.
pub struct PerChannelParams {
pub:
	scales      []f64
	zero_points []int
	axis        int
	qmin        i8 = -128
	qmax        i8 = 127
}

// new_params validates and creates signed INT8 affine quantization parameters.
pub fn new_params(scale f64, zero_point int, qmin i8, qmax i8) !QuantizationParams {
	validate_range(qmin, qmax)!
	if !math.is_finite(scale) || scale <= 0 {
		return error('quantization scale must be finite and positive')
	}
	if zero_point < int(qmin) || zero_point > int(qmax) {
		return error('zero point ${zero_point} is outside [${qmin}, ${qmax}]')
	}
	return QuantizationParams{
		scale:      scale
		zero_point: zero_point
		qmin:       qmin
		qmax:       qmax
	}
}

// affine_params derives asymmetric affine parameters from a finite value range.
pub fn affine_params(minimum f64, maximum f64, qmin i8, qmax i8) !QuantizationParams {
	validate_range(qmin, qmax)!
	if !math.is_finite(minimum) || !math.is_finite(maximum) || minimum > maximum {
		return error('calibration range must be finite and minimum <= maximum')
	}
	if qmin > 0 || qmax < 0 {
		return error('signed affine quantization range must include zero')
	}
	if minimum == maximum {
		if minimum == 0 {
			return new_params(1, 0, qmin, qmax)
		}
		endpoint := if minimum > 0 { int(qmax) } else { -int(qmin) }
		return new_params(math.abs(minimum) / f64(endpoint), 0, qmin, qmax)
	}
	calibration_minimum := math.min(minimum, 0)
	calibration_maximum := math.max(maximum, 0)
	span := f64(int(qmax) - int(qmin))
	// Divide first to avoid overflowing when the observed range spans most of
	// the f64 domain.
	scale := calibration_maximum / span - calibration_minimum / span
	if !math.is_finite(scale) || scale <= 0 {
		return error('calibration range cannot be represented by a finite scale')
	}
	mut zero := math.round_to_even(f64(int(qmin)) - calibration_minimum / scale)
	if zero < f64(int(qmin)) {
		zero = f64(int(qmin))
	} else if zero > f64(int(qmax)) {
		zero = f64(int(qmax))
	}
	return new_params(scale, int(zero), qmin, qmax)
}

// new_per_channel_params validates per-channel INT8 affine parameters.
pub fn new_per_channel_params(scales []f64, zero_points []int, axis int, qmin i8, qmax i8) !PerChannelParams {
	validate_range(qmin, qmax)!
	if scales.len == 0 || scales.len != zero_points.len {
		return error('per-channel scales and zero points must have the same non-zero length')
	}
	for i, scale in scales {
		if !math.is_finite(scale) || scale <= 0 {
			return error('per-channel scale ${i} must be finite and positive')
		}
		if zero_points[i] < int(qmin) || zero_points[i] > int(qmax) {
			return error('per-channel zero point ${i} is outside [${qmin}, ${qmax}]')
		}
	}
	return PerChannelParams{
		scales:      scales.clone()
		zero_points: zero_points.clone()
		axis:        axis
		qmin:        qmin
		qmax:        qmax
	}
}

// quantize_per_tensor_f32 maps finite f32 values into signed INT8 values.
pub fn quantize_per_tensor_f32(input &vtl.Tensor[f32], params QuantizationParams) !&vtl.Tensor[i8] {
	validate_params(params)!
	values := quantize_values(f32_values(input), params)!
	return vtl.from_array[i8](values, input.shape)!
}

// quantize_per_tensor_f64 maps finite f64 values into signed INT8 values.
pub fn quantize_per_tensor_f64(input &vtl.Tensor[f64], params QuantizationParams) !&vtl.Tensor[i8] {
	validate_params(params)!
	values := quantize_values(input.to_array(), params)!
	return vtl.from_array[i8](values, input.shape)!
}

// dequantize_per_tensor_f32 converts signed INT8 values to f32 values.
pub fn dequantize_per_tensor_f32(input &vtl.Tensor[i8], params QuantizationParams) !&vtl.Tensor[f32] {
	validate_params(params)!
	return vtl.from_array[f32](f32_values_from_dequantized(input.to_array(), params.scale,
		params.zero_point), input.shape)!
}

// dequantize_per_tensor_f64 converts signed INT8 values to f64 values.
pub fn dequantize_per_tensor_f64(input &vtl.Tensor[i8], params QuantizationParams) !&vtl.Tensor[f64] {
	validate_params(params)!
	return vtl.from_array[f64](dequantize_values(input.to_array(), params.scale, params.zero_point),
		input.shape)!
}

// quantize_per_channel_f32 applies one scale and zero point per selected channel.
pub fn quantize_per_channel_f32(input &vtl.Tensor[f32], params PerChannelParams) !&vtl.Tensor[i8] {
	values := quantize_channel_values(f32_values(input), input.shape, params)!
	return vtl.from_array[i8](values, input.shape)!
}

// quantize_per_channel_f64 applies one scale and zero point per selected channel.
pub fn quantize_per_channel_f64(input &vtl.Tensor[f64], params PerChannelParams) !&vtl.Tensor[i8] {
	values := quantize_channel_values(input.to_array(), input.shape, params)!
	return vtl.from_array[i8](values, input.shape)!
}

// dequantize_per_channel_f32 converts signed INT8 values to f32 values.
pub fn dequantize_per_channel_f32(input &vtl.Tensor[i8], params PerChannelParams) !&vtl.Tensor[f32] {
	values := dequantize_channel_values(input.to_array(), input.shape, params)!
	return vtl.from_array[f32](f32_values_from_f64(values), input.shape)!
}

// dequantize_per_channel_f64 converts signed INT8 values to f64 values.
pub fn dequantize_per_channel_f64(input &vtl.Tensor[i8], params PerChannelParams) !&vtl.Tensor[f64] {
	return vtl.from_array[f64](dequantize_channel_values(input.to_array(), input.shape, params)!,
		input.shape)!
}

// MinMaxObserver accumulates finite scalar range statistics for calibration.
pub struct MinMaxObserver {
pub mut:
	minimum     f64
	maximum     f64
	initialized bool
}

// observe_f32 adds finite f32 tensor values to the calibration range.
pub fn (mut observer MinMaxObserver) observe_f32(input &vtl.Tensor[f32]) ! {
	update_minmax(mut observer, f32_values(input))!
}

// observe_f64 adds finite f64 tensor values to the calibration range.
pub fn (mut observer MinMaxObserver) observe_f64(input &vtl.Tensor[f64]) ! {
	update_minmax(mut observer, input.to_array())!
}

// params derives affine quantization parameters from observed values.
pub fn (observer &MinMaxObserver) params(qmin i8, qmax i8) !QuantizationParams {
	if !observer.initialized {
		return error('observer has no calibration values')
	}
	return affine_params(observer.minimum, observer.maximum, qmin, qmax)
}

// PerChannelMinMaxObserver accumulates independent ranges along one tensor axis.
pub struct PerChannelMinMaxObserver {
pub:
	axis int
pub mut:
	minimums    []f64
	maximums    []f64
	initialized []bool
}

// new_per_channel_min_max_observer creates a per-channel calibration observer.
pub fn new_per_channel_min_max_observer(axis int) PerChannelMinMaxObserver {
	return PerChannelMinMaxObserver{
		axis: axis
	}
}

// observe_f32 adds finite f32 values to matching channel ranges.
pub fn (mut observer PerChannelMinMaxObserver) observe_f32(input &vtl.Tensor[f32]) ! {
	update_channel_minmax(mut observer, f32_values(input), input.shape)!
}

// observe_f64 adds finite f64 values to matching channel ranges.
pub fn (mut observer PerChannelMinMaxObserver) observe_f64(input &vtl.Tensor[f64]) ! {
	update_channel_minmax(mut observer, input.to_array(), input.shape)!
}

// params derives one affine parameter set per observed channel.
pub fn (observer &PerChannelMinMaxObserver) params(qmin i8, qmax i8) !PerChannelParams {
	if observer.minimums.len == 0 {
		return error('observer has no calibration values')
	}
	mut scales := []f64{len: observer.minimums.len}
	mut zero_points := []int{len: observer.minimums.len}
	for i in 0 .. observer.minimums.len {
		if !observer.initialized[i] {
			return error('observer channel ${i} has no calibration values')
		}
		p := affine_params(observer.minimums[i], observer.maximums[i], qmin, qmax)!
		scales[i] = p.scale
		zero_points[i] = p.zero_point
	}
	return new_per_channel_params(scales, zero_points, observer.axis, qmin, qmax)
}

fn update_minmax(mut observer MinMaxObserver, values []f64) ! {
	for i, value in values {
		if !math.is_finite(value) {
			return error('observer cannot consume non-finite value at logical index ${i}')
		}
	}
	for value in values {
		if !observer.initialized {
			observer.minimum = value
			observer.maximum = value
			observer.initialized = true
		} else {
			observer.minimum = math.min(observer.minimum, value)
			observer.maximum = math.max(observer.maximum, value)
		}
	}
}

fn update_channel_minmax(mut observer PerChannelMinMaxObserver, values []f64, shape []int) ! {
	axis := normalize_axis(observer.axis, shape.len)!
	channels := shape[axis]
	if channels == 0 {
		return error('cannot calibrate an empty channel axis')
	}
	for i, value in values {
		if !math.is_finite(value) {
			return error('observer cannot consume non-finite value at logical index ${i}')
		}
	}
	if observer.minimums.len == 0 {
		observer.minimums = []f64{len: channels}
		observer.maximums = []f64{len: channels}
		observer.initialized = []bool{len: channels}
	} else if observer.minimums.len != channels {
		return error('channel count changed from ${observer.minimums.len} to ${channels}')
	}
	trailing := trailing_size(shape, axis)
	for i, value in values {
		channel := (i / trailing) % channels
		if !observer.initialized[channel] {
			observer.minimums[channel] = value
			observer.maximums[channel] = value
			observer.initialized[channel] = true
		} else {
			observer.minimums[channel] = math.min(observer.minimums[channel], value)
			observer.maximums[channel] = math.max(observer.maximums[channel], value)
		}
	}
}

fn quantize_values(values []f64, params QuantizationParams) ![]i8 {
	mut output := []i8{len: values.len}
	for i, value in values {
		if !math.is_finite(value) {
			return error('cannot quantize non-finite value at logical index ${i}')
		}
		output[i] = quantize_value(value, params.scale, params.zero_point, params.qmin, params.qmax)
	}
	return output
}

fn quantize_channel_values(values []f64, shape []int, params PerChannelParams) ![]i8 {
	axis := normalize_axis(params.axis, shape.len)!
	validate_channel_params(params, shape[axis])!
	mut output := []i8{len: values.len}
	trailing := trailing_size(shape, axis)
	for i, value in values {
		if !math.is_finite(value) {
			return error('cannot quantize non-finite value at logical index ${i}')
		}
		channel := (i / trailing) % shape[axis]
		output[i] = quantize_value(value, params.scales[channel], params.zero_points[channel],
			params.qmin, params.qmax)
	}
	return output
}

fn dequantize_values(values []i8, scale f64, zero_point int) []f64 {
	mut output := []f64{len: values.len}
	for i, value in values {
		output[i] = (f64(value) - f64(zero_point)) * scale
	}
	return output
}

fn dequantize_channel_values(values []i8, shape []int, params PerChannelParams) ![]f64 {
	axis := normalize_axis(params.axis, shape.len)!
	validate_channel_params(params, shape[axis])!
	mut output := []f64{len: values.len}
	trailing := trailing_size(shape, axis)
	for i, value in values {
		channel := (i / trailing) % shape[axis]
		output[i] = (f64(value) - f64(params.zero_points[channel])) * params.scales[channel]
	}
	return output
}

fn f32_values(input &vtl.Tensor[f32]) []f64 {
	values := input.to_array()
	mut output := []f64{len: values.len}
	for i, value in values {
		output[i] = f64(value)
	}
	return output
}

fn f32_values_from_f64(values []f64) []f32 {
	mut output := []f32{len: values.len}
	for i, value in values {
		output[i] = f32(value)
	}
	return output
}

fn f32_values_from_dequantized(values []i8, scale f64, zero_point int) []f32 {
	return f32_values_from_f64(dequantize_values(values, scale, zero_point))
}

fn validate_params(params QuantizationParams) ! {
	_ := new_params(params.scale, params.zero_point, params.qmin, params.qmax)!
}

fn validate_channel_params(params PerChannelParams, channels int) ! {
	if params.scales.len != channels {
		return error('expected ${channels} per-channel parameters, got ${params.scales.len}')
	}
	_ := new_per_channel_params(params.scales, params.zero_points, params.axis, params.qmin,
		params.qmax)!
}

fn validate_range(qmin i8, qmax i8) ! {
	if qmin >= qmax {
		return error('quantization qmin must be less than qmax')
	}
}

fn normalize_axis(axis int, rank int) !int {
	if rank == 0 {
		return error('per-channel quantization requires a non-scalar tensor')
	}
	resolved := if axis < 0 { axis + rank } else { axis }
	if resolved < 0 || resolved >= rank {
		return error('axis ${axis} is out of range for rank ${rank}')
	}
	return resolved
}

fn trailing_size(shape []int, axis int) int {
	mut size := 1
	for i in axis + 1 .. shape.len {
		size *= shape[i]
	}
	return size
}

fn quantize_value(value f64, scale f64, zero_point int, qmin i8, qmax i8) i8 {
	scaled := value / scale + f64(zero_point)
	if scaled <= f64(int(qmin)) - 1 {
		return qmin
	}
	if scaled >= f64(int(qmax)) + 1 {
		return qmax
	}
	rounded := math.round_to_even(scaled)
	if rounded < f64(int(qmin)) {
		return qmin
	}
	if rounded > f64(int(qmax)) {
		return qmax
	}
	return i8(int(rounded))
}
