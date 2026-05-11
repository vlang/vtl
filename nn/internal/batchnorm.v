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
			normalized := (f64(input.get([n, c])) - batch_mean_data[c]) / math.sqrt(
				batch_var_data[c] + eps)
			output_data[n * num_features + c] = f64(gamma.get([0, c])) * normalized +
				f64(beta.get([0, c]))
		}
	}
	output := vtl.from_array(output_data.map(vtl.cast[T](it)), [batch_size, num_features])!
	return output, batch_mean, batch_var
}

// batchnorm1d_backward computes gradients w.r.t. input, gamma, beta.
pub fn batchnorm1d_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], mean &vtl.Tensor[T], var_ &vtl.Tensor[T], eps f64) ![]&vtl.Tensor[T] {
	batch_size := input.shape[0]
	num_features := input.shape[1]

	// dL/dx = gamma * (grad_out - mean(grad_out)) / sqrt(var+eps)
	// First compute mean of gradient per feature
	mut grad_mean_data := []f64{len: num_features}
	for c in 0 .. num_features {
		mut sum := f64(0)
		for n in 0 .. batch_size {
			sum += f64(gradient.get([n, c]))
		}
		grad_mean_data[c] = sum / f64(batch_size)
	}

	// std = sqrt(var + eps)
	mut std_data := []f64{len: num_features}
	for c in 0 .. num_features {
		std_data[c] = math.sqrt(f64(var_.get([0, c])) + eps)
	}

	// dL/dx per element
	mut dx_data := []f64{len: batch_size * num_features}
	for n in 0 .. batch_size {
		for c in 0 .. num_features {
			dx_data[n * num_features + c] = f64(gamma.get([0, c])) * (f64(gradient.get([
				n,
				c,
			])) - grad_mean_data[c]) / std_data[c]
		}
	}
	dx := vtl.from_array(dx_data.map(vtl.cast[T](it)), [batch_size, num_features])!

	// dL/dgamma = sum over batch of grad_out * normalized_x
	mut dgamma_data := []f64{len: num_features}
	for c in 0 .. num_features {
		mut sum := f64(0)
		mean_c := f64(mean.get([0, c]))
		std_c := std_data[c]
		for n in 0 .. batch_size {
			normalized := (f64(input.get([n, c])) - mean_c) / std_c
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
