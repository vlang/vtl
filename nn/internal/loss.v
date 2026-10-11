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

// nll computes mean Negative Log Likelihood from log-probabilities. Targets can
// be class indices `[batch]` or one-hot/soft labels `[batch, classes]`.
pub fn nll[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.shape.len != 2 || input.shape[0] == 0 || input.shape[1] == 0 {
		return error('NLL loss requires non-empty rank-two log-probabilities')
	}
	batch_size := input.shape[0]
	n_classes := input.shape[1]
	class_indices := target.shape.len == 1 && target.shape[0] == batch_size
	if !class_indices && target.shape != input.shape {
		return error('NLL targets must match log-probabilities or have shape [batch]')
	}
	mut total := 0.0
	for i in 0 .. batch_size {
		if class_indices {
			class_index := nll_class_index[T](target, i, n_classes)!
			total -= f64(input.get([i, class_index]))
		} else {
			for c in 0 .. n_classes {
				total -= f64(target.get([i, c])) * f64(input.get([i, c]))
			}
		}
	}
	return vtl.from_1d([vtl.cast[T](total / f64(batch_size))])
}

// nll_backward computes gradient of NLL w.r.t. input (log-probs)
pub fn nll_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.shape.len != 2 || input.shape[0] == 0 || input.shape[1] == 0 {
		return error('NLL loss requires non-empty rank-two log-probabilities')
	}
	batch_size := input.shape[0]
	n_classes := input.shape[1]
	class_indices := target.shape.len == 1 && target.shape[0] == batch_size
	if !class_indices && target.shape != input.shape {
		return error('NLL targets must match log-probabilities or have shape [batch]')
	}
	upstream := f64(gradient.get([0]))
	mut ret_data := []f64{len: batch_size * n_classes}
	for i in 0 .. batch_size {
		mut class_index := -1
		if class_indices {
			class_index = nll_class_index[T](target, i, n_classes)!
		}
		for c in 0 .. n_classes {
			tgt := if class_indices {
				if c == class_index { 1.0 } else { 0.0 }
			} else {
				f64(target.get([i, c]))
			}
			ret_data[i * n_classes + c] = upstream * -tgt / f64(batch_size)
		}
	}
	ret := vtl.from_array(ret_data.map(vtl.cast[T](it)), input.shape.clone())!
	return ret
}

fn nll_class_index[T](target &vtl.Tensor[T], row int, n_classes int) !int {
	raw_class := f64(target.get_nth(row))
	if !math.is_finite(raw_class) {
		return error('NLL class indices must be finite integers')
	}
	class_index := int(raw_class)
	if raw_class != f64(class_index) || class_index < 0 || class_index >= n_classes {
		return error('NLL class index is out of range or not an integer')
	}
	return class_index
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

// cross_entropy computes CrossEntropyLoss (LogSoftmax + NLL combined).
// This is the standard cross-entropy for multi-class classification.
// input: [batch, n_classes] raw logits
// target: one-hot/soft labels `[batch, n_classes]` or class indices `[batch]`.
pub fn cross_entropy[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	// Forward: log_softmax(x) = x - log(sum(exp(x - max)))
	//          loss = -mean(sum(target * log_softmax(x)))
	if input.shape.len != 2 || input.shape[0] == 0 || input.shape[1] == 0 {
		return error('Cross entropy requires non-empty rank-two logits')
	}
	batch_size := input.shape[0]
	n_classes := input.shape[1]
	class_indices := target.shape.len == 1 && target.shape[0] == batch_size
	if !class_indices && target.shape != input.shape {
		return error('Cross entropy targets must match logits or have shape [batch]')
	}
	mut total_loss := f64(0)
	for i in 0 .. batch_size {
		mut max_logit := f64(input.get([i, 0]))
		for c in 1 .. n_classes {
			v := f64(input.get([i, c]))
			if v > max_logit { max_logit = v }
		}
		mut sum_exp := f64(0)
		for c in 0 .. n_classes {
			sum_exp += math.exp(f64(input.get([i, c])) - max_logit)
		}
		log_sum_exp := math.log(sum_exp) + max_logit
		if class_indices {
			class_index := cross_entropy_class_index[T](target, i, n_classes)!
			total_loss += log_sum_exp - f64(input.get([i, class_index]))
		} else {
			mut dot := f64(0)
			for c in 0 .. n_classes {
				dot += f64(target.get([i, c])) * (f64(input.get([i, c])) - log_sum_exp)
			}
			total_loss -= dot
		}
	}
	return vtl.from_1d([vtl.cast[T](total_loss / f64(batch_size))])
}

fn cross_entropy_class_index[T](target &vtl.Tensor[T], row int, n_classes int) !int {
	raw_class := f64(target.get_nth(row))
	if !math.is_finite(raw_class) {
		return error('Cross entropy class indices must be finite integers')
	}
	class_index := int(raw_class)
	if raw_class != f64(class_index) || class_index < 0 || class_index >= n_classes {
		return error('Cross entropy class index is out of range or not an integer')
	}
	return class_index
}

// cross_entropy_backward computes gradient of CrossEntropyLoss w.r.t. input logits.
// dL/dx_i = (sum(target) * softmax(x)_i - target_i) / batch_size
pub fn cross_entropy_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if input.shape.len != 2 || input.shape[0] == 0 || input.shape[1] == 0 {
		return error('Cross entropy requires non-empty rank-two logits')
	}
	batch_size := input.shape[0]
	n_classes := input.shape[1]
	class_indices := target.shape.len == 1 && target.shape[0] == batch_size
	if !class_indices && target.shape != input.shape {
		return error('Cross entropy targets must match logits or have shape [batch]')
	}
	upstream := f64(gradient.get([0]))
	mut ret_data := []f64{len: batch_size * n_classes}
	for i in 0 .. batch_size {
		mut max_logit := f64(input.get([i, 0]))
		for c in 1 .. n_classes {
			v := f64(input.get([i, c]))
			if v > max_logit { max_logit = v }
		}
		mut sum_exp := f64(0)
		for c in 0 .. n_classes {
			sum_exp += math.exp(f64(input.get([i, c])) - max_logit)
		}
		mut target_sum := 0.0
		mut class_index := -1
		if class_indices {
			class_index = cross_entropy_class_index[T](target, i, n_classes)!
			target_sum = 1.0
		} else {
			for c in 0 .. n_classes {
				target_sum += f64(target.get([i, c]))
			}
		}
		for c in 0 .. n_classes {
			sm := math.exp(f64(input.get([i, c])) - max_logit) / sum_exp
			tgt := if class_indices {
				if c == class_index { 1.0 } else { 0.0 }
			} else {
				f64(target.get([i, c]))
			}
			ret_data[i * n_classes + c] = upstream * (sm * target_sum - tgt) / f64(batch_size)
		}
	}
	return vtl.from_array(ret_data.map(vtl.cast[T](it)), [batch_size, n_classes])
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
