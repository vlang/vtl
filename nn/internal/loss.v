module internal

import math
import vtl

// mse_backward exposes this operation as part of the public API.
pub fn mse_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	input_size := cache.size()
	ret := gradient.nmap([cache, target], fn [input_size] [T](vals []T, i []int) T {
		return vals[0] * vtl.cast[T](2) * (vals[1] - vals[2]) / vtl.cast[T](input_size)
	})!
	return [ret]
}

// sigmoid_cross_entropy_backward exposes this operation as part of the public API.
pub fn sigmoid_cross_entropy_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	batch_size := cache.shape[0]
	// Reshape target from [batch_size] → [batch_size, 1] so shapes align with cache [batch_size, 1].
	// Without this, broadcasting [4,1] × [4] produces [4,4] instead of [4,1].
	target_2d := target.reshape([batch_size, 1])!
	// Iterate over cache (the logits, shape [batch_size, 1]).
	// vals[0] = logit, vals[1] = target label, vals[2] = upstream gradient (scalar, broadcast from [1]).
	// Gradient of sigmoid-BCE w.r.t. logit: upstream * (sigmoid(logit) - target) / batch_size
	ret := cache.nmap([target_2d, gradient], fn [batch_size] [T](vals []T, i []int) T {
		logit := vals[0]
		tgt := vals[1]
		upstream := vals[2]
		sigma := vtl.cast[T](1) / (vtl.cast[T](1) + vtl.cast[T](math.exp(-f64(logit))))
		return upstream * (sigma - tgt) / vtl.cast[T](batch_size)
	})!
	return [ret]
}

// softmax_cross_entropy_backward computes the gradient of the SCE loss w.r.t.
// the logits. The gradient for each logit is:
//   upstream * (sum(target) * softmax(logit_i) - target_i) / batch_size
//
// This is the standard gradient derived from the log-softmax formulation
// (equivalent to Arraymancer's implementation).
pub fn softmax_cross_entropy_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	batch_size := cache.shape[0]
	n_classes := cache.shape[1]
	upstream := f64(gradient.get([0]))
	mut ret_data := []f64{len: batch_size * n_classes}
	for i in 0 .. batch_size {
		// Compute softmax for sample i
		mut max_logit := f64(cache.get([i, 0]))
		for c in 1 .. n_classes {
			v := f64(cache.get([i, c]))
			if v > max_logit {
				max_logit = v
			}
		}
		mut sum_exp := f64(0)
		for c in 0 .. n_classes {
			sum_exp += math.exp(f64(cache.get([i, c])) - max_logit)
		}
		mut target_sum := 0.0
		for c in 0 .. n_classes {
			target_sum += f64(target.get([i, c]))
		}
		for c in 0 .. n_classes {
			sm := math.exp(f64(cache.get([i, c])) - max_logit) / sum_exp
			tgt := f64(target.get([i, c]))
			ret_data[i * n_classes + c] = upstream * (sm * target_sum - tgt) / f64(batch_size)
		}
	}
	ret := vtl.from_array(ret_data.map(vtl.cast[T](it)), [batch_size, n_classes])!
	return [ret]
}

// bce_with_logits computes mean binary cross entropy directly from logits using
// the stable softplus form: max(x, 0) - x*y + log(1 + exp(-abs(x))).
@[direct_array_access; inline]
pub fn bce_with_logits[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.shape != target.shape {
		return error('BCE loss requires input and target with identical shapes')
	}
	if input.shape == target.shape && input.is_row_major_contiguous()
		&& target.is_row_major_contiguous() && input.data.data.len == input.size()
		&& target.data.data.len == target.size() {
		mut sum := vtl.cast[T](0)
		for i in 0 .. input.size() {
			x := f64(input.data.data[i])
			y := f64(target.data.data[i])
			if !math.is_finite(y) || y < 0.0 || y > 1.0 {
				return error('BCE loss targets must be in [0, 1]')
			}
			value := math.max(x, 0.0) - x * y + math.log(1.0 + math.exp(-math.abs(x)))
			sum += vtl.cast[T](value)
		}
		return vtl.from_1d([sum / vtl.cast[T](input.size())])
	}
	mut sum := vtl.cast[T](0)
	for i in 0 .. input.size() {
		x := f64(input.get_nth(i))
		y := f64(target.get_nth(i))
		if !math.is_finite(y) || y < 0.0 || y > 1.0 {
			return error('BCE loss targets must be in [0, 1]')
		}
		value := math.max(x, 0.0) - x * y + math.log(1.0 + math.exp(-math.abs(x)))
		sum += vtl.cast[T](value)
	}
	return vtl.from_1d([sum / vtl.cast[T](input.size())])
}

// bce computes mean binary cross entropy from probabilities in [0, 1]. Log
// outputs are floored at -100 to keep boundary cases finite.

// bce exposes this operation as part of the public API.

// bce exposes this operation as part of the public API.
@[direct_array_access; inline]
pub fn bce[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.shape != target.shape {
		return error('BCE loss requires input and target with identical shapes')
	}
	if input.is_row_major_contiguous()
		&& target.is_row_major_contiguous() && input.data.data.len == input.size()
		&& target.data.data.len == target.size() {
		mut sum := vtl.cast[T](0)
		for i in 0 .. input.size() {
			probability := f64(input.data.data[i])
			y := f64(target.data.data[i])
			if !math.is_finite(probability) || probability < 0.0 || probability > 1.0 {
				return error('BCE loss probabilities must be in [0, 1]')
			}
			if !math.is_finite(y) || y < 0.0 || y > 1.0 {
				return error('BCE loss targets must be in [0, 1]')
			}
			log_probability := if probability == 0.0 {
				-100.0
			} else {
				math.max(math.log(probability), -100.0)
			}
			log_one_minus_probability := if probability == 1.0 {
				-100.0
			} else {
				math.max(math.log(1.0 - probability), -100.0)
			}
			sum += vtl.cast[T](-y * log_probability - (1.0 - y) * log_one_minus_probability)
		}
		return vtl.from_1d([sum / vtl.cast[T](input.size())])
	}
	mut sum := vtl.cast[T](0)
	for i in 0 .. input.size() {
		probability := f64(input.get_nth(i))
		y := f64(target.get_nth(i))
		if !math.is_finite(probability) || probability < 0.0 || probability > 1.0 {
			return error('BCE loss probabilities must be in [0, 1]')
		}
		if !math.is_finite(y) || y < 0.0 || y > 1.0 {
			return error('BCE loss targets must be in [0, 1]')
		}
		log_probability := if probability == 0.0 {
			-100.0
		} else {
			math.max(math.log(probability), -100.0)
		}
		log_one_minus_probability := if probability == 1.0 {
			-100.0
		} else {
			math.max(math.log(1.0 - probability), -100.0)
		}
		sum += vtl.cast[T](-y * log_probability - (1.0 - y) * log_one_minus_probability)
	}
	return vtl.from_1d([sum / vtl.cast[T](input.size())])
}

// bce_backward computes the gradient of mean BCE w.r.t. its input.
// Logit inputs use the simplified sigmoid(logit) - target derivative; probability
// inputs use the BCELoss derivative with PyTorch's denominator floor.
pub fn bce_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T], from_logits bool) !&vtl.Tensor[T] {
	input_size := input.size()
	upstream := f64(gradient.get([0]))
	mut ret_data := []T{len: input_size}
	for i in 0 .. input_size {
		x := f64(input.get_nth(i))
		t := f64(target.get_nth(i))
		mut grad := 0.0
		if from_logits {
			p := 1.0 / (1.0 + math.exp(-x))
			grad = (p - t) / f64(input_size)
		} else {
			denominator := math.max(x * (1.0 - x), 1e-12)
			grad = ((x - t) / denominator) / f64(input_size)
		}
		ret_data[i] = vtl.cast[T](upstream * grad)
	}

	ret_shape := input.shape.clone()
	ret := vtl.from_array(ret_data, ret_shape)!
	return ret
}

// huber computes the Huber loss (smooth L1)
pub fn huber[T](input &vtl.Tensor[T], target &vtl.Tensor[T], delta T) !&vtl.Tensor[T] {
	half := vtl.cast[T](0.5)
	mut total := vtl.cast[T](0)
	for i in 0 .. input.size() {
		diff := math.abs(vtl.cast[f64](input.get_nth(i) - target.get_nth(i)))
		if diff > vtl.cast[f64](delta) {
			total += delta * vtl.cast[T](diff - 0.5 * vtl.cast[f64](delta))
		} else {
			diff_f := vtl.cast[f64](input.get_nth(i) - target.get_nth(i))
			total += half * vtl.cast[T](diff_f * diff_f)
		}
	}
	return vtl.from_1d([total / vtl.cast[T](input.size())])
}

// huber_backward computes the gradient of Huber loss w.r.t. input
pub fn huber_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T], delta T) !&vtl.Tensor[T] {
	batch_size := input.size()
	upstream := f64(gradient.get([0]))
	mut ret_data := []f64{len: batch_size}
	for i in 0 .. batch_size {
		diff := f64(input.get_nth(i) - target.get_nth(i))
		abs_diff := math.abs(diff)
		mut grad := f64(0)
		if abs_diff > vtl.cast[f64](delta) {
			grad = if diff > 0 { 1.0 } else { -1.0 } * vtl.cast[f64](delta)
		} else {
			grad = diff
		}
		ret_data[i] = upstream * grad / f64(batch_size)
	}
	ret := vtl.from_array(ret_data.map(vtl.cast[T](it)), input.shape.clone())!
	return ret
}

// smooth_l1 computes PyTorch-style Smooth L1 with the selected reduction.
pub fn smooth_l1[T](input &vtl.Tensor[T], target &vtl.Tensor[T], beta T, reduction string) !&vtl.Tensor[T] {
	if input.shape != target.shape {
		return error('Smooth L1 loss requires input and target with identical shapes')
	}
	if reduction !in ['none', 'mean', 'sum'] {
		return error('Smooth L1 reduction must be none, mean, or sum')
	}
	beta_value := f64(beta)
	if !math.is_finite(beta_value) || beta_value < 0.0 {
		return error('Smooth L1 beta must be finite and non-negative')
	}
	mut total := 0.0
	mut values := []T{len: input.size()}
	for i in 0 .. input.size() {
		difference := f64(input.get_nth(i)) - f64(target.get_nth(i))
		absolute_difference := math.abs(difference)
		value := if beta_value == 0.0 {
			absolute_difference
		} else if absolute_difference < beta_value {
			0.5 * difference * difference / beta_value
		} else {
			absolute_difference - 0.5 * beta_value
		}
		values[i] = vtl.cast[T](value)
		total += value
	}
	if reduction == 'none' {
		return vtl.from_array(values, input.shape.clone())
	}
	if reduction == 'sum' {
		return vtl.from_1d([vtl.cast[T](total)])
	}
	if input.size() == 0 {
		return vtl.from_1d([vtl.cast[T](math.nan())])
	}
	return vtl.from_1d([vtl.cast[T](total / f64(input.size()))])
}

// smooth_l1_backward computes the gradient of Smooth L1 for the selected reduction.
pub fn smooth_l1_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T], beta T, reduction string) !&vtl.Tensor[T] {
	if input.shape != target.shape {
		return error('Smooth L1 loss requires input and target with identical shapes')
	}
	if reduction !in ['none', 'mean', 'sum'] {
		return error('Smooth L1 reduction must be none, mean, or sum')
	}
	beta_value := f64(beta)
	if !math.is_finite(beta_value) || beta_value < 0.0 {
		return error('Smooth L1 beta must be finite and non-negative')
	}
	mut values := []T{len: input.size()}
	for i in 0 .. input.size() {
		difference := f64(input.get_nth(i)) - f64(target.get_nth(i))
		derivative := if beta_value == 0.0 {
			if difference > 0.0 {
				1.0
			} else if difference < 0.0 {
				-1.0
			} else {
				difference
			}
		} else if math.abs(difference) < beta_value {
			difference / beta_value
		} else if difference > 0.0 {
			1.0
		} else if difference < 0.0 {
			-1.0
		} else {
			difference
		}
		upstream := if reduction == 'none' {
			f64(gradient.get_nth(i))
		} else {
			f64(gradient.get([0]))
		}
		normalization := if reduction == 'mean' && input.size() > 0 {
			f64(input.size())
		} else {
			1.0
		}
		values[i] = vtl.cast[T](upstream * derivative / normalization)
	}
	return vtl.from_array(values, input.shape.clone())
}

// nll computes Negative Log Likelihood from log-probabilities. Class-index
// targets omit the class axis; one-hot/soft targets match the input shape.
pub fn nll[T](input &vtl.Tensor[T], target &vtl.Tensor[T], weight &vtl.Tensor[T], ignore_index int, reduction string) !&vtl.Tensor[T] {
	if input.shape.len < 1 || input.shape[0] == 0 || input.size() == 0 {
		return error('NLL loss requires non-empty log-probabilities with a class axis')
	}
	class_axis := nll_class_axis(input.shape)
	n_classes := input.shape[class_axis]
	index_shape := nll_index_shape(input.shape)
	sample_count := nll_sample_count(input.shape)
	if reduction !in ['none', 'mean', 'sum'] {
		return error('NLL reduction must be none, mean, or sum')
	}
	if !isnil(weight) && (weight.shape.len != 1 || weight.shape[0] != n_classes) {
		return error('NLL class weights must have shape [n_classes]')
	}
	class_indices := target.shape == index_shape
	if !class_indices && target.shape != input.shape {
		return error('NLL target shape must omit the class axis or match log-probabilities')
	}
	if !class_indices && ignore_index != -1 {
		return error('NLL ignore_index is supported only for class-index targets')
	}
	mut total := 0.0
	mut weight_total := 0.0
	mut valid_targets := 0
	mut per_sample := []f64{len: sample_count}
	for i in 0 .. sample_count {
		if class_indices {
			raw_class := f64(target.get_nth(i))
			if !math.is_finite(raw_class) || raw_class != math.floor(raw_class) {
				return error('NLL class indices must be finite integers')
			}
			if raw_class == f64(ignore_index) {
				continue
			}
			class_index := nll_class_index[T](target, i, n_classes)!
			class_weight := if isnil(weight) { 1.0 } else { f64(weight.get_nth(class_index)) }
			coords := nll_sample_coordinates(input.shape, i, class_index)
			per_sample[i] = -class_weight * f64(input.get(coords))
			total += per_sample[i]
			weight_total += class_weight
			valid_targets++
		} else {
			for c in 0 .. n_classes {
				class_weight := if isnil(weight) { 1.0 } else { f64(weight.get_nth(c)) }
				coords := nll_sample_coordinates(input.shape, i, c)
				per_sample[i] -= f64(target.get(coords)) * class_weight * f64(input.get(coords))
			}
			total += per_sample[i]
		}
	}
	if reduction == 'none' {
		return vtl.from_array(per_sample.map(vtl.cast[T](it)), index_shape)
	}
	denominator := if reduction == 'sum' {
		1.0
	} else if class_indices && !isnil(weight) {
		weight_total
	} else if class_indices {
		f64(valid_targets)
	} else {
		f64(sample_count)
	}
	if denominator == 0.0 {
		if valid_targets == 0 && class_indices {
			return nll_reduced_result[T](vtl.cast[T](0), input.shape)
		}
		return error('NLL mean reduction requires non-zero total target weight')
	}
	return nll_reduced_result[T](vtl.cast[T](total / denominator), input.shape)
}

// nll_backward computes the gradient of NLL w.r.t. input (log-probabilities).
pub fn nll_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T], weight &vtl.Tensor[T], ignore_index int, reduction string) !&vtl.Tensor[T] {
	if input.shape.len < 1 || input.shape[0] == 0 || input.size() == 0 {
		return error('NLL loss requires non-empty log-probabilities with a class axis')
	}
	class_axis := nll_class_axis(input.shape)
	n_classes := input.shape[class_axis]
	index_shape := nll_index_shape(input.shape)
	sample_count := nll_sample_count(input.shape)
	if reduction !in ['none', 'mean', 'sum'] {
		return error('NLL reduction must be none, mean, or sum')
	}
	if !isnil(weight) && (weight.shape.len != 1 || weight.shape[0] != n_classes) {
		return error('NLL class weights must have shape [n_classes]')
	}
	class_indices := target.shape == index_shape
	if !class_indices && target.shape != input.shape {
		return error('NLL target shape must omit the class axis or match log-probabilities')
	}
	if !class_indices && ignore_index != -1 {
		return error('NLL ignore_index is supported only for class-index targets')
	}
	mut denominator := if reduction in ['none', 'sum'] { 1.0 } else { f64(sample_count) }
	if reduction == 'mean' && class_indices {
		denominator = 0.0
		for i in 0 .. sample_count {
			raw_class := f64(target.get_nth(i))
			if !math.is_finite(raw_class) || raw_class != math.floor(raw_class) {
				return error('NLL class indices must be finite integers')
			}
			if raw_class == f64(ignore_index) {
				continue
			}
			class_index := nll_class_index[T](target, i, n_classes)!
			denominator += if isnil(weight) { 1.0 } else { f64(weight.get_nth(class_index)) }
		}
	}
	mut ret_data := []f64{len: input.size()}
	for i in 0 .. sample_count {
		mut class_index := -1
		if class_indices {
			raw_class := f64(target.get_nth(i))
			if !math.is_finite(raw_class) || raw_class != math.floor(raw_class) {
				return error('NLL class indices must be finite integers')
			}
			if raw_class == f64(ignore_index) {
				continue
			}
			class_index = nll_class_index[T](target, i, n_classes)!
		}
		for c in 0 .. n_classes {
			coords := nll_sample_coordinates(input.shape, i, c)
			tgt := if class_indices {
				if c == class_index { 1.0 } else { 0.0 }
			} else {
				f64(target.get(coords))
			}
			class_weight := if isnil(weight) { 1.0 } else { f64(weight.get_nth(c)) }
			upstream := if reduction == 'none' {
				f64(gradient.get_nth(i))
			} else {
				f64(gradient.get_nth(0))
			}
			ret_data[nll_sample_flat_index(input.shape, i, c)] = upstream * -tgt * class_weight / denominator
		}
	}
	ret := vtl.from_array(ret_data.map(vtl.cast[T](it)), input.shape.clone())!
	return ret
}

fn nll_sample_count(input_shape []int) int {
	if input_shape.len == 1 {
		return 1
	}
	mut count := input_shape[0]
	for axis in 2 .. input_shape.len {
		count *= input_shape[axis]
	}
	return count
}

fn nll_sample_coordinates(input_shape []int, sample_index int, class_index int) []int {
	if input_shape.len == 1 {
		return [class_index]
	}
	spatial_size := nll_sample_count(input_shape) / input_shape[0]
	mut coords := []int{len: input_shape.len}
	coords[0] = sample_index / spatial_size
	coords[1] = class_index
	mut position := sample_index % spatial_size
	for axis in 2 .. input_shape.len {
		mut stride := 1
		for trailing_axis in axis + 1 .. input_shape.len {
			stride *= input_shape[trailing_axis]
		}
		coords[axis] = position / stride
		position %= stride
	}
	return coords
}

fn nll_sample_flat_index(input_shape []int, sample_index int, class_index int) int {
	if input_shape.len == 1 {
		return class_index
	}
	spatial_size := nll_sample_count(input_shape) / input_shape[0]
	batch_index := sample_index / spatial_size
	position := sample_index % spatial_size
	return (batch_index * input_shape[1] + class_index) * spatial_size + position
}

fn nll_class_axis(input_shape []int) int {
	return if input_shape.len == 1 { 0 } else { 1 }
}

fn nll_index_shape(input_shape []int) []int {
	if input_shape.len == 1 {
		return []int{}
	}
	mut shape := [input_shape[0]]
	shape << input_shape[2..]
	return shape
}

fn nll_reduced_result[T](value T, input_shape []int) !&vtl.Tensor[T] {
	if input_shape.len == 1 {
		return vtl.from_array([value], [])
	}
	return vtl.from_1d([value])
}

fn nll_class_index[T](target &vtl.Tensor[T], row int, n_classes int) !int {
	raw_class := f64(target.get_nth(row))
	if !math.is_finite(raw_class) {
		return error('NLL class indices must be finite integers')
	}
	if raw_class != math.floor(raw_class) || raw_class < 0.0 || raw_class >= f64(n_classes) {
		return error('NLL class index is out of range or not an integer')
	}
	return int(raw_class)
}

// kl_div computes KL Divergence loss D_KL(P || Q) = sum(P * log(P/Q))
// input: log-probs Q, target: probabilities P
pub fn kl_div[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	mut total := vtl.cast[T](0)
	for i in 0 .. input.size() {
		p := f64(target.get_nth(i))
		q := f64(input.get_nth(i))
		if p > 0 {
			total += vtl.cast[T](p * math.log(p / math.exp(q)))
		}
	}
	return vtl.from_1d([total / vtl.cast[T](input.size())])
}

// kl_div_backward computes gradient of KL Divergence w.r.t. input (log-probs Q)
pub fn kl_div_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	input_size := input.size()
	upstream := f64(gradient.get([0]))
	mut ret_data := []f64{len: input_size}
	for i in 0 .. input_size {
		p := f64(target.get_nth(i))
		ret_data[i] = upstream * -p / f64(input_size)
	}
	ret := vtl.from_array(ret_data.map(vtl.cast[T](it)), input.shape.clone())!
	return ret
}

// cross_entropy computes mean CrossEntropyLoss from raw logits.
pub fn cross_entropy[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return cross_entropy_with_options[T](input, target, unsafe { nil }, -100, 'mean', 0)
}

// cross_entropy_with_options computes weighted CrossEntropyLoss with configurable
// ignored class indices and none, mean, or sum reduction.
pub fn cross_entropy_with_options[T](input &vtl.Tensor[T], target &vtl.Tensor[T], weight &vtl.Tensor[T], ignore_index int, reduction string, label_smoothing f64) !&vtl.Tensor[T] {
	log_probs := cross_entropy_log_probs[T](input)!
	cross_entropy_validate_target[T](input, target, ignore_index)!
	if !math.is_finite(label_smoothing) || label_smoothing < 0 || label_smoothing > 1 {
		return error('Cross entropy label_smoothing must be finite and between 0 and 1')
	}
	if label_smoothing > 0 {
		return cross_entropy_smoothed[T](log_probs, target, weight, ignore_index, reduction,
			label_smoothing)
	}
	// PyTorch ignores ignore_index for probability targets; NLL's lower-level
	// contract requires the sentinel value -1 when targets are not class indices.
	index_shape := nll_index_shape(input.shape)
	nll_ignore_index := if target.shape == index_shape { ignore_index } else { -1 }
	return nll[T](log_probs, target, weight, nll_ignore_index, reduction)
}

fn cross_entropy_smoothed[T](log_probs &vtl.Tensor[T], target &vtl.Tensor[T], weight &vtl.Tensor[T], ignore_index int, reduction string, label_smoothing f64) !&vtl.Tensor[T] {
	if reduction !in ['none', 'mean', 'sum'] {
		return error('Cross entropy reduction must be none, mean, or sum')
	}
	input_shape := log_probs.shape
	class_axis := nll_class_axis(input_shape)
	n_classes := input_shape[class_axis]
	if !isnil(weight) && (weight.shape.len != 1 || weight.shape[0] != n_classes) {
		return error('Cross entropy class weights must have shape [n_classes]')
	}
	index_shape := nll_index_shape(input_shape)
	class_indices := target.shape == index_shape
	sample_count := nll_sample_count(input_shape)
	mut denominator := f64(sample_count)
	mut valid_targets := 0
	if class_indices {
		denominator = 0
		for sample in 0 .. sample_count {
			target_class := int(target.get_nth(sample))
			if target_class == ignore_index {
				continue
			}
			valid_targets++
			denominator += if isnil(weight) { 1.0 } else { f64(weight.get_nth(target_class)) }
		}
	}
	mut total := f64(0)
	mut per_sample := []T{len: sample_count}
	for sample in 0 .. sample_count {
		if class_indices && int(target.get_nth(sample)) == ignore_index {
			continue
		}
		target_class := if class_indices { int(target.get_nth(sample)) } else { -1 }
		mut sample_loss := f64(0)
		for class_index in 0 .. n_classes {
			target_probability := cross_entropy_smoothed_target_probability[T](target, input_shape,
				class_indices, sample, class_index, target_class, n_classes, label_smoothing)
			class_weight := if isnil(weight) { 1.0 } else { f64(weight.get_nth(class_index)) }
			log_probability := f64(log_probs.get_nth(nll_sample_flat_index(input_shape, sample,
				class_index)))
			sample_loss -= target_probability * class_weight * log_probability
		}
		per_sample[sample] = vtl.cast[T](sample_loss)
		total += sample_loss
	}
	if reduction == 'none' {
		return vtl.from_array(per_sample, index_shape.clone())
	}
	if reduction == 'sum' {
		return nll_reduced_result[T](vtl.cast[T](total), input_shape)
	}
	if denominator == 0 {
		if class_indices && valid_targets == 0 {
			return nll_reduced_result[T](vtl.cast[T](0), input_shape)
		}
		return error('Cross entropy mean reduction requires non-zero total target weight')
	}
	return nll_reduced_result[T](vtl.cast[T](total / denominator), input_shape)
}

fn cross_entropy_smoothed_target_probability[T](target &vtl.Tensor[T], input_shape []int, class_indices bool, sample int, class_index int, target_class int, n_classes int, label_smoothing f64) f64 {
	target_probability := if class_indices {
		if class_index == target_class { 1.0 } else { 0.0 }
	} else {
		f64(target.get(nll_sample_coordinates(input_shape, sample, class_index)))
	}
	return (1.0 - label_smoothing) * target_probability + label_smoothing / f64(n_classes)
}

// cross_entropy_backward computes the mean CrossEntropyLoss gradient w.r.t. logits.
pub fn cross_entropy_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return cross_entropy_backward_with_options[T](gradient, input, target, unsafe { nil }, -100,
		'mean', 0)
}

// cross_entropy_backward_with_options applies the NLL gradient to log-softmax
// and then its Jacobian to produce the gradient with respect to raw logits.
pub fn cross_entropy_backward_with_options[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T], weight &vtl.Tensor[T], ignore_index int, reduction string, label_smoothing f64) !&vtl.Tensor[T] {
	log_probs := cross_entropy_log_probs[T](input)!
	cross_entropy_validate_target[T](input, target, ignore_index)!
	if !math.is_finite(label_smoothing) || label_smoothing < 0 || label_smoothing > 1 {
		return error('Cross entropy label_smoothing must be finite and between 0 and 1')
	}
	if label_smoothing > 0 {
		return cross_entropy_smoothed_backward[T](gradient, input, log_probs, target, weight,
			ignore_index, reduction, label_smoothing)
	}
	class_axis := nll_class_axis(input.shape)
	index_shape := nll_index_shape(input.shape)
	nll_ignore_index := if target.shape == index_shape { ignore_index } else { -1 }
	log_prob_gradient := nll_backward[T](gradient, log_probs, target, weight, nll_ignore_index,
		reduction)!
	n_classes := input.shape[class_axis]
	sample_count := nll_sample_count(input.shape)
	mut result := []T{len: input.size()}
	for sample in 0 .. sample_count {
		mut gradient_sum := f64(0)
		for class_index in 0 .. n_classes {
			flat_index := nll_sample_flat_index(input.shape, sample, class_index)
			gradient_sum += f64(log_prob_gradient.get_nth(flat_index))
		}
		for class_index in 0 .. n_classes {
			flat_index := nll_sample_flat_index(input.shape, sample, class_index)
			probability := math.exp(f64(log_probs.get_nth(flat_index)))
			result[flat_index] = vtl.cast[T](f64(log_prob_gradient.get_nth(flat_index)) - probability * gradient_sum)
		}
	}
	return vtl.from_array(result, input.shape.clone())
}

fn cross_entropy_smoothed_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], log_probs &vtl.Tensor[T], target &vtl.Tensor[T], weight &vtl.Tensor[T], ignore_index int, reduction string, label_smoothing f64) !&vtl.Tensor[T] {
	if reduction !in ['none', 'mean', 'sum'] {
		return error('Cross entropy reduction must be none, mean, or sum')
	}
	class_axis := nll_class_axis(input.shape)
	n_classes := input.shape[class_axis]
	if !isnil(weight) && (weight.shape.len != 1 || weight.shape[0] != n_classes) {
		return error('Cross entropy class weights must have shape [n_classes]')
	}
	index_shape := nll_index_shape(input.shape)
	class_indices := target.shape == index_shape
	sample_count := nll_sample_count(input.shape)
	mut denominator := f64(sample_count)
	if class_indices {
		denominator = 0
		for sample in 0 .. sample_count {
			target_class := int(target.get_nth(sample))
			if target_class == ignore_index {
				continue
			}
			denominator += if isnil(weight) { 1.0 } else { f64(weight.get_nth(target_class)) }
		}
	}
	mut result := []T{len: input.size()}
	for sample in 0 .. sample_count {
		if class_indices && int(target.get_nth(sample)) == ignore_index {
			continue
		}
		target_class := if class_indices { int(target.get_nth(sample)) } else { -1 }
		mut weighted_target_sum := f64(0)
		for class_index in 0 .. n_classes {
			target_probability := cross_entropy_smoothed_target_probability[T](target, input.shape,
				class_indices, sample, class_index, target_class, n_classes, label_smoothing)
			class_weight := if isnil(weight) { 1.0 } else { f64(weight.get_nth(class_index)) }
			weighted_target_sum += target_probability * class_weight
		}
		upstream := if reduction == 'none' {
			f64(gradient.get_nth(sample))
		} else {
			f64(gradient.get_nth(0))
		}
		reduction_scale := if reduction == 'mean' && denominator > 0 {
			upstream / denominator
		} else {
			upstream
		}
		for class_index in 0 .. n_classes {
			flat_index := nll_sample_flat_index(input.shape, sample, class_index)
			target_probability := cross_entropy_smoothed_target_probability[T](target, input.shape,
				class_indices, sample, class_index, target_class, n_classes, label_smoothing)
			class_weight := if isnil(weight) { 1.0 } else { f64(weight.get_nth(class_index)) }
			probability := math.exp(f64(log_probs.get_nth(flat_index)))
			result[flat_index] = vtl.cast[T](reduction_scale * (probability * weighted_target_sum - target_probability * class_weight))
		}
	}
	return vtl.from_array(result, input.shape.clone())
}

fn cross_entropy_validate_target[T](input &vtl.Tensor[T], target &vtl.Tensor[T], ignore_index int) ! {
	index_shape := nll_index_shape(input.shape)
	if target.shape != index_shape && target.shape != input.shape {
		return error('Cross entropy targets must match logits or omit the class axis')
	}
	if target.shape != index_shape {
		return
	}
	n_classes := input.shape[nll_class_axis(input.shape)]
	for i in 0 .. target.size() {
		raw_class := f64(target.get_nth(i))
		if raw_class == f64(ignore_index) {
			continue
		}
		if !math.is_finite(raw_class) || raw_class != math.floor(raw_class) {
			return error('Cross entropy class index must be a finite integer')
		}
		if raw_class < 0 || raw_class >= f64(n_classes) {
			return error('Cross entropy class index is out of range')
		}
	}
}

fn cross_entropy_log_probs[T](input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.shape.len == 0 || input.size() == 0 {
		return error('Cross entropy requires non-empty logits with a class axis')
	}
	class_axis := nll_class_axis(input.shape)
	n_classes := input.shape[class_axis]
	if n_classes == 0 {
		return error('Cross entropy requires at least one class')
	}
	sample_count := nll_sample_count(input.shape)
	mut result := []T{len: input.size()}
	for sample in 0 .. sample_count {
		mut max_logit := f64(input.get(nll_sample_coordinates(input.shape, sample, 0)))
		for class_index in 1 .. n_classes {
			logit := f64(input.get(nll_sample_coordinates(input.shape, sample, class_index)))
			if logit > max_logit {
				max_logit = logit
			}
		}
		mut sum_exp := f64(0)
		for class_index in 0 .. n_classes {
			logit := f64(input.get(nll_sample_coordinates(input.shape, sample, class_index)))
			sum_exp += math.exp(logit - max_logit)
		}
		log_sum_exp := math.log(sum_exp) + max_logit
		for class_index in 0 .. n_classes {
			flat_index := nll_sample_flat_index(input.shape, sample, class_index)
			logit := f64(input.get(nll_sample_coordinates(input.shape, sample, class_index)))
			result[flat_index] = vtl.cast[T](logit - log_sum_exp)
		}
	}
	return vtl.from_array(result, input.shape.clone())
}

// l1 computes mean absolute error and returns a scalar tensor.
pub fn l1[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.shape != target.shape {
		return error('L1 loss requires input and target with identical shapes')
	}
	if input.size() == 0 { return error('L1 loss requires non-empty tensors') }
	mut total := f64(0)
	for i in 0 .. input.size() { total += math.abs(f64(input.get_nth(i)) - f64(target.get_nth(i))) }
	return vtl.from_1d([vtl.cast[T](total / f64(input.size()))])
}

// l1_backward computes the mean absolute error gradient; the subgradient at zero is zero.
pub fn l1_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.shape != target.shape {
		return error('L1 loss requires input and target with identical shapes')
	}
	if input.size() == 0 { return error('L1 loss requires non-empty tensors') }
	upstream := f64(gradient.get([0]))
	n := f64(input.size())
	mut values := []T{len: input.size()}
	for i in 0 .. input.size() {
		diff := f64(input.get_nth(i)) - f64(target.get_nth(i))
		sign := if diff > 0 {
			1.0
		} else if diff < 0 {
			-1.0
		} else {
			0.0
		}
		values[i] = vtl.cast[T](upstream * sign / n)
	}
	return vtl.from_array(values, input.shape.clone())
}

// hinge computes mean max(0, 1 - target * score); labels must be -1 or +1.
pub fn hinge[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.shape != target.shape {
		return error('Hinge loss requires input and target with identical shapes')
	}
	if input.size() == 0 { return error('Hinge loss requires non-empty tensors') }
	mut total := f64(0)
	for i in 0 .. input.size() {
		ty := f64(target.get_nth(i))
		if ty != -1.0 && ty != 1.0 { return error('Hinge loss labels must be -1 or +1') }
		total += math.max(0.0, 1.0 - ty * f64(input.get_nth(i)))
	}
	return vtl.from_1d([vtl.cast[T](total / f64(input.size()))])
}

// hinge_backward computes the gradient of mean hinge loss.
pub fn hinge_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.shape != target.shape {
		return error('Hinge loss requires input and target with identical shapes')
	}
	if input.size() == 0 { return error('Hinge loss requires non-empty tensors') }
	upstream := f64(gradient.get([0])) / f64(input.size())
	mut values := []T{len: input.size()}
	for i in 0 .. input.size() {
		ty := f64(target.get_nth(i))
		if ty != -1.0 && ty != 1.0 { return error('Hinge loss labels must be -1 or +1') }
		grad := if 1.0 - ty * f64(input.get_nth(i)) > 0.0 { -ty * upstream } else { 0.0 }
		values[i] = vtl.cast[T](grad)
	}
	return vtl.from_array(values, input.shape.clone())
}

// focal computes mean binary focal loss. When from_logits is true input contains logits.
pub fn focal[T](input &vtl.Tensor[T], target &vtl.Tensor[T], alpha f64, gamma f64, from_logits bool) !&vtl.Tensor[T] {
	if input.shape != target.shape {
		return error('Focal loss requires input and target with identical shapes')
	}
	if input.size() == 0 { return error('Focal loss requires non-empty tensors') }
	if !math.is_finite(alpha) || !math.is_finite(gamma) || alpha < 0.0 || alpha > 1.0 || gamma < 0.0 {
		return error('Focal loss requires finite alpha in [0, 1] and finite gamma >= 0')
	}
	mut total := f64(0)
	for i in 0 .. input.size() {
		y := f64(target.get_nth(i))
		if y != 0.0 && y != 1.0 { return error('Focal loss targets must be 0 or +1') }
		x := f64(input.get_nth(i))
		if !from_logits && (x < 0.0 || x > 1.0) {
			return error('Focal loss probabilities must be in [0, 1]')
		}
		mut p := if from_logits {
			if x >= 0 {
				1.0 / (1.0 + math.exp(-x))
			} else {
				ex := math.exp(x)
				ex / (1.0 + ex)
			}
		} else {
			x
		}
		p = math.max(1e-15, math.min(1.0 - 1e-15, p))
		pt := if y == 1.0 { p } else { 1.0 - p }
		alpha_t := if y == 1.0 { alpha } else { 1.0 - alpha }
		total -= alpha_t * math.pow(1.0 - pt, gamma) * math.log(pt)
	}
	return vtl.from_1d([vtl.cast[T](total / f64(input.size()))])
}

// focal_backward computes the gradient of mean binary focal loss.
pub fn focal_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T], alpha f64, gamma f64, from_logits bool) !&vtl.Tensor[T] {
	if input.shape != target.shape {
		return error('Focal loss requires input and target with identical shapes')
	}
	if input.size() == 0 { return error('Focal loss requires non-empty tensors') }
	if !math.is_finite(alpha) || !math.is_finite(gamma) || alpha < 0.0 || alpha > 1.0 || gamma < 0.0 {
		return error('Focal loss requires finite alpha in [0, 1] and finite gamma >= 0')
	}
	upstream := f64(gradient.get([0])) / f64(input.size())
	mut values := []T{len: input.size()}
	for i in 0 .. input.size() {
		y := f64(target.get_nth(i))
		if y != 0.0 && y != 1.0 { return error('Focal loss targets must be 0 or +1') }
		x := f64(input.get_nth(i))
		if !from_logits && (x < 0.0 || x > 1.0) {
			return error('Focal loss probabilities must be in [0, 1]')
		}
		mut p := if from_logits {
			if x >= 0 {
				1.0 / (1.0 + math.exp(-x))
			} else {
				ex := math.exp(x)
				ex / (1.0 + ex)
			}
		} else {
			x
		}
		p = math.max(1e-15, math.min(1.0 - 1e-15, p))
		pt := if y == 1.0 { p } else { 1.0 - p }
		alpha_t := if y == 1.0 { alpha } else { 1.0 - alpha }
		modulation := math.pow(1.0 - pt, gamma)
		dloss_dpt := alpha_t * (gamma * math.pow(1.0 - pt, gamma - 1.0) * math.log(pt) - modulation / pt)
		mut dloss_dp := if y == 1.0 { dloss_dpt } else { -dloss_dpt }
		if from_logits { dloss_dp *= p * (1.0 - p) }
		values[i] = vtl.cast[T](upstream * dloss_dp)
	}
	return vtl.from_array(values, input.shape.clone())
}
