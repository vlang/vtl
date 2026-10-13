module internal

import math
import vtl

// batchnorm1d_forward computes batch norm using running mean/var (inference path).
pub fn batchnorm1d_forward[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], running_mean &vtl.Tensor[T], running_var &vtl.Tensor[T], eps f64) !&vtl.Tensor[T] {
	std := running_var.map(fn [eps] [T](val T, i []int) T {
		return vtl.cast[T](1.0 / (math.sqrt(f64(val) + eps)))
	})
	// (x - running_mean) * gamma / std + beta
	centered := input.map(fn [running_mean] [T](val T, i []int) T {
		return val - running_mean.get([0, i[1]])
	})
	normalized := centered.nmap([std], fn [T](vals []T, i []int) T {
		return vals[0] * vals[1]
	})!
	output := normalized.nmap([gamma, beta], fn [T](vals []T, i []int) T {
		return vals[0] * vals[1] + vals[2]
	})!
	return output
}

// batchnorm1d_training computes batch norm using batch stats (training path).
pub fn batchnorm1d_training[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) !(&vtl.Tensor[T], &vtl.Tensor[T], &vtl.Tensor[T]) {
	batch_size := input.shape[0]
	num_features := input.shape[1]

	// Compute batch mean: mean over batch dimension (axis=0)
	mut batch_mean_data := []f64{len: num_features}
	for c in 0 .. num_features {
		mut sum := f64(0)
		for n in 0 .. batch_size {
			sum += f64(input.get([n, c]))
		}
		batch_mean_data[c] = sum / f64(batch_size)
	}
	batch_mean := vtl.from_array(batch_mean_data.map(vtl.cast[T](it)), [1, num_features])!

	// Compute batch variance
	mut batch_var_data := []f64{len: num_features}
	for c in 0 .. num_features {
		mut sum := f64(0)
		mean_c := batch_mean_data[c]
		for n in 0 .. batch_size {
			diff := f64(input.get([n, c])) - mean_c
			sum += diff * diff
		}
		batch_var_data[c] = sum / f64(batch_size)
	}
	batch_var := vtl.from_array(batch_var_data.map(vtl.cast[T](it)), [1, num_features])!

	// Normalize: (x - mean) / sqrt(var + eps)
	mut output_data := []f64{len: batch_size * num_features}
	for n in 0 .. batch_size {
		for c in 0 .. num_features {
			normalized := (f64(input.get([n, c])) - batch_mean_data[c]) / math.sqrt(batch_var_data[c] + eps)
			output_data[n * num_features + c] = f64(gamma.get([0, c])) * normalized +
				f64(beta.get([0, c]))
		}
	}
	output := vtl.from_array(output_data.map(vtl.cast[T](it)), [batch_size, num_features])!
	return output, batch_mean, batch_var
}

// batchnorm2d_forward computes BatchNorm2D using running mean/var
// (inference path). Input is [N, C, H, W]; gamma, beta and the running
// statistics are per-channel vectors of length C.
pub fn batchnorm2d_forward[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], running_mean &vtl.Tensor[T], running_var &vtl.Tensor[T], eps f64) !&vtl.Tensor[T] {
	n_size := input.shape[0]
	channels := input.shape[1]
	height := input.shape[2]
	width := input.shape[3]
	mut output_data := []f64{len: n_size * channels * height * width}
	for n in 0 .. n_size {
		for c in 0 .. channels {
			inv_std := 1.0 / math.sqrt(f64(running_var.get([c])) + eps)
			mean_c := f64(running_mean.get([c]))
			gamma_c := f64(gamma.get([c]))
			beta_c := f64(beta.get([c]))
			for h in 0 .. height {
				for w in 0 .. width {
					idx := ((n * channels + c) * height + h) * width + w
					output_data[idx] = gamma_c * (f64(input.get([n, c, h, w])) - mean_c) * inv_std + beta_c
				}
			}
		}
	}
	return vtl.from_array(output_data.map(vtl.cast[T](it)), input.shape)!
}

// batchnorm2d_training computes BatchNorm2D using per-channel batch
// statistics over N*H*W elements (training path). Returns the output plus
// the biased batch mean and variance as [C] tensors.
pub fn batchnorm2d_training[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) !(&vtl.Tensor[T], &vtl.Tensor[T], &vtl.Tensor[T]) {
	n_size := input.shape[0]
	channels := input.shape[1]
	height := input.shape[2]
	width := input.shape[3]
	count := f64(n_size * height * width)

	mut mean_data := []f64{len: channels}
	for c in 0 .. channels {
		mut sum := f64(0)
		for n in 0 .. n_size {
			for h in 0 .. height {
				for w in 0 .. width {
					sum += f64(input.get([n, c, h, w]))
				}
			}
		}
		mean_data[c] = sum / count
	}

	mut var_data := []f64{len: channels}
	for c in 0 .. channels {
		mut sum := f64(0)
		for n in 0 .. n_size {
			for h in 0 .. height {
				for w in 0 .. width {
					diff := f64(input.get([n, c, h, w])) - mean_data[c]
					sum += diff * diff
				}
			}
		}
		var_data[c] = sum / count
	}

	mut output_data := []f64{len: n_size * channels * height * width}
	for n in 0 .. n_size {
		for c in 0 .. channels {
			inv_std := 1.0 / math.sqrt(var_data[c] + eps)
			gamma_c := f64(gamma.get([c]))
			beta_c := f64(beta.get([c]))
			for h in 0 .. height {
				for w in 0 .. width {
					idx := ((n * channels + c) * height + h) * width + w
					output_data[idx] = gamma_c * (f64(input.get([n, c, h, w])) - mean_data[c]) * inv_std + beta_c
				}
			}
		}
	}
	output := vtl.from_array(output_data.map(vtl.cast[T](it)), input.shape)!
	batch_mean := vtl.from_array(mean_data.map(vtl.cast[T](it)), [channels])!
	batch_var := vtl.from_array(var_data.map(vtl.cast[T](it)), [channels])!
	return output, batch_mean, batch_var
}

// batchnorm2d_backward_mode computes gradients w.r.t. input, gamma and beta
// for BatchNorm2D, using batch statistics in training mode and the fixed
// running statistics as a constant affine transform in evaluation mode.
pub fn batchnorm2d_backward_mode[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], gamma &vtl.Tensor[T], _beta &vtl.Tensor[T], mean &vtl.Tensor[T], var_ &vtl.Tensor[T], eps f64, training bool) ![]&vtl.Tensor[T] {
	n_size := input.shape[0]
	channels := input.shape[1]
	height := input.shape[2]
	width := input.shape[3]
	count := f64(n_size * height * width)

	mut dx_data := []f64{len: n_size * channels * height * width}
	mut dgamma_data := []f64{len: channels}
	mut dbeta_data := []f64{len: channels}
	for c in 0 .. channels {
		std_c := math.sqrt(f64(var_.get([c])) + eps)
		mean_c := f64(mean.get([c]))
		gamma_c := f64(gamma.get([c]))
		mut grad_sum := f64(0)
		mut grad_norm_sum := f64(0)
		for n in 0 .. n_size {
			for h in 0 .. height {
				for w in 0 .. width {
					grad := f64(gradient.get([n, c, h, w]))
					normalized := (f64(input.get([n, c, h, w])) - mean_c) / std_c
					grad_sum += grad
					grad_norm_sum += grad * normalized
				}
			}
		}
		dgamma_data[c] = grad_norm_sum
		dbeta_data[c] = grad_sum
		for n in 0 .. n_size {
			for h in 0 .. height {
				for w in 0 .. width {
					grad := f64(gradient.get([n, c, h, w]))
					mut dx := grad
					if training {
						normalized := (f64(input.get([n, c, h, w])) - mean_c) / std_c
						dx = (count * grad - grad_sum - normalized * grad_norm_sum) / count
					}
					idx := ((n * channels + c) * height + h) * width + w
					dx_data[idx] = gamma_c * dx / std_c
				}
			}
		}
	}
	dx := vtl.from_array(dx_data.map(vtl.cast[T](it)), input.shape)!
	dgamma := vtl.from_array(dgamma_data.map(vtl.cast[T](it)), [channels])!
	dbeta := vtl.from_array(dbeta_data.map(vtl.cast[T](it)), [channels])!
	return [dx, dgamma, dbeta]
}

// batchnorm1d_backward computes gradients w.r.t. input, gamma, beta.
pub fn batchnorm1d_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], mean &vtl.Tensor[T], var_ &vtl.Tensor[T], eps f64) ![]&vtl.Tensor[T] {
	return batchnorm1d_backward_mode[T](gradient, input, gamma, beta, mean, var_, eps, true)
}

// batchnorm1d_backward_mode computes gradients using batch or fixed running
// statistics, depending on whether the layer is training.
pub fn batchnorm1d_backward_mode[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], gamma &vtl.Tensor[T], _beta &vtl.Tensor[T], mean &vtl.Tensor[T], var_ &vtl.Tensor[T], eps f64, training bool) ![]&vtl.Tensor[T] {
	batch_size := input.shape[0]
	num_features := input.shape[1]

	// dL/dx uses the batch-normalization Jacobian in training and a fixed
	// affine transform in evaluation.
	mut std_data := []f64{len: num_features}
	mut grad_sum_data := []f64{len: num_features}
	mut grad_norm_sum_data := []f64{len: num_features}
	for c in 0 .. num_features {
		std_data[c] = math.sqrt(f64(var_.get([0, c])) + eps)
		mean_c := f64(mean.get([0, c]))
		mut grad_sum := f64(0)
		mut grad_norm_sum := f64(0)
		for n in 0 .. batch_size {
			grad := f64(gradient.get([n, c]))
			normalized := (f64(input.get([n, c])) - mean_c) / std_data[c]
			grad_sum += grad
			grad_norm_sum += grad * normalized
		}
		grad_sum_data[c] = grad_sum
		grad_norm_sum_data[c] = grad_norm_sum
	}

	// dL/dx per element
	mut dx_data := []f64{len: batch_size * num_features}
	for n in 0 .. batch_size {
		for c in 0 .. num_features {
			grad := f64(gradient.get([n, c]))
			if training {
				mean_c := f64(mean.get([0, c]))
				normalized := (f64(input.get([n, c])) - mean_c) / std_data[c]
				grad = (f64(batch_size) * grad - grad_sum_data[c] - normalized * grad_norm_sum_data[c]) / f64(batch_size)
			}
			dx_data[n * num_features + c] = f64(gamma.get([0, c])) * grad / std_data[c]
		}
	}
	dx := vtl.from_array(dx_data.map(vtl.cast[T](it)), [batch_size, num_features])!

	// dL/dgamma = sum over batch of grad_out * normalized_x
	mut dgamma_data := []f64{len: num_features}
	for c in 0 .. num_features {
		mut sum := f64(0)
		std_c := std_data[c]
		for n in 0 .. batch_size {
			normalized := (f64(input.get([n, c])) - f64(mean.get([0, c]))) / std_c
			sum += f64(gradient.get([n, c])) * normalized
		}
		dgamma_data[c] = sum
	}
	dgamma := vtl.from_array(dgamma_data.map(vtl.cast[T](it)), [1, num_features])!

	// dL/dbeta = sum over batch of grad_out
	mut dbeta_data := []f64{len: num_features}
	for c in 0 .. num_features {
		mut sum := f64(0)
		for n in 0 .. batch_size {
			sum += f64(gradient.get([n, c]))
		}
		dbeta_data[c] = sum
	}
	dbeta := vtl.from_array(dbeta_data.map(vtl.cast[T](it)), [1, num_features])!

	return [dx, dgamma, dbeta]
}
