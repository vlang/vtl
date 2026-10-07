module stats

import vtl
import math
import math.complex as cmplx

// VarianceData configures the degrees of freedom used by variance and std.
pub struct VarianceData {
pub:
	ddof int
}

// variance calculates the variance using Welford's stable online algorithm.
// It returns f64 even for integer tensors to preserve fractional results.
// ddof is subtracted from the number of observations in the denominator;
// the default (0) computes population variance.
pub fn variance[T](t &vtl.Tensor[T], data VarianceData) !f64 {
	if data.ddof < 0 {
		return error('variance: ddof must be non-negative')
	}
	if t.size <= data.ddof {
		return error('variance: observations must exceed ddof')
	}
	mut count := 0
	mut average := 0.0
	mut sum_squares := 0.0
	mut iter := t.iterator()
	for {
		value, _ := iter.next() or { break }
		count++
		delta := f64(value) - average
		average += delta / f64(count)
		sum_squares += delta * (f64(value) - average)
	}
	return sum_squares / f64(count - data.ddof)
}

// std calculates the standard deviation using Welford's stable algorithm.
pub fn std[T](t &vtl.Tensor[T], data VarianceData) !f64 {
	return math.sqrt(variance(t, data)?)
}

// AxisData defines a public data structure for this module.
pub struct AxisData {
pub:
	axis int
}

// sum returns the sum of all elements of the given tensor
@[direct_array_access]
pub fn sum[T](t &vtl.Tensor[T]) T {
	if is_flat_tensor_storage(t) {
		data := t.data.data
		mut total0 := sum_identity[T]()
		mut total1 := sum_identity[T]()
		mut total2 := sum_identity[T]()
		mut total3 := sum_identity[T]()
		mut total4 := sum_identity[T]()
		mut total5 := sum_identity[T]()
		mut total6 := sum_identity[T]()
		mut total7 := sum_identity[T]()
		mut index := 0
		for index + 7 < data.len {
			total0 += data[index]
			total1 += data[index + 1]
			total2 += data[index + 2]
			total3 += data[index + 3]
			total4 += data[index + 4]
			total5 += data[index + 5]
			total6 += data[index + 6]
			total7 += data[index + 7]
			index += 8
		}
		mut total := (((total0 + total1) + (total2 + total3)) + (total4 + total5)) + (total6 + total7)
		for index < data.len {
			total += data[index]
			index++
		}
		return total
	}
	return t.reduce(sum_identity[T](), fn [T](acc T, val T, i []int) T {
		return acc + val
	})
}

fn is_flat_tensor_storage[T](t &vtl.Tensor[T]) bool {
	return t.data.data.len == t.size && t.is_row_major_contiguous()
}

// sum_axis returns the sum of a given Tensor along a provided
// axis
pub fn sum_axis[T](t &vtl.Tensor[T], data AxisData) T {
	mut iter := t.axis_iterator(data.axis)
	mut acc := sum_identity[T]()
	for {
		val, _ := iter.next() or { break }
		acc += val
	}
	return acc
}

// sum_axis_dims returns the sum of a given Tensor along a provided
// axis with the reduced dimension intact
pub fn sum_axis_with_dims[T](t &vtl.Tensor[T], data AxisData) T {
	mut iter := t.axis_with_dims_iterator(data.axis)
	mut acc := sum_identity[T]()
	for {
		val, _ := iter.next() or { break }
		acc += val
	}
	return acc
}

// prod returns the product of all elements of the given tensor
@[direct_array_access]
pub fn prod[T](t &vtl.Tensor[T]) T {
	if is_flat_tensor_storage(t) {
		mut product := product_identity[T]()
		for value in t.data.data {
			product *= value
		}
		return product
	}
	return t.reduce(product_identity[T](), fn [T](acc T, val T, i []int) T {
		return acc * val
	})
}

// prod_axis_dims returns the product of a given Tensor along a provided
// axis with the reduced dimension intact
pub fn prod_axis[T](t &vtl.Tensor[T], data AxisData) T {
	mut iter := t.axis_iterator(data.axis)
	mut acc := product_identity[T]()
	for {
		val, _ := iter.next() or { break }
		acc *= val
	}
	return acc
}

// prod_axis_dims returns the product of a Tensor along a provided
// axis with the reduced dimension intact
pub fn prod_axis_with_dims[T](t &vtl.Tensor[T], data AxisData) T {
	mut iter := t.axis_with_dims_iterator(data.axis)
	mut acc := product_identity[T]()
	for {
		val, _ := iter.next() or { break }
		acc *= val
	}
	return acc
}

// Measure of Occurrence
// Frequency of a given number
// Based on
// https://www.mathsisfun.com/data/frequency-distribution.html
pub fn freq[T](t &vtl.Tensor[T], val T) int {
	if t.size == 0 {
		return 0
	}

	mut iter := t.iterator()
	mut count := 0
	for {
		v, _ := iter.next() or { break }
		if v == val {
			count += 1
		}
	}
	return count
}

// Measure of Central Tendency
// Mean of the given input array
// Based on
// https://www.mathsisfun.com/data/central-measures.html
pub fn mean[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		$if T is cmplx.Complex {
			return T(cmplx.Complex{ re: math.nan(), im: math.nan() })
		} $else {
			return sum_identity[T]()
		}
	}
	$if T is cmplx.Complex {
		return sum(t) / T(cmplx.Complex{ re: f64(t.size), im: 0 })
	} $else {
		return sum(t) / vtl.cast[T](t.size)
	}
}

fn sum_identity[T]() T {
	$if T is cmplx.Complex {
		return T(cmplx.Complex{ re: 0, im: 0 })
	} $else {
		return vtl.cast[T](0)
	}
}

fn product_identity[T]() T {
	$if T is cmplx.Complex {
		return T(cmplx.Complex{ re: 1, im: 0 })
	} $else {
		return vtl.cast[T](1)
	}
}

// Measure of Central Tendency
// Geometric Mean of the given input array
// Based on
// https://www.mathsisfun.com/numbers/geometric-mean.html
pub fn geometric_mean[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	prod := t.reduce(vtl.cast[T](1.0), fn [T](acc T, val T, i []int) T {
		return acc * val
	})
	return math.pow(prod, vtl.cast[T](1) / vtl.cast[T](t.size))
}

// Measure of Central Tendency
// Harmonic Mean of the given input array
// Based on
// https://www.mathsisfun.com/numbers/harmonic-mean.html
pub fn harmonic_mean[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	return vtl.cast[T](t.size) / t.reduce(vtl.cast[T](0), fn [T](acc T, val T, i []int) T {
		return acc + (vtl.cast[T](1) / val)
	})
}

// median returns the linearly interpolated median of the tensor values.
// Input values can be in any order; integer tensors return an f64 so even-sized
// inputs can have fractional medians. Empty inputs return NaN.
pub fn median[T](t &vtl.Tensor[T]) f64 {
	return quantile_linear[T](t, 0.5) or { math.nan() }
}

// Measure of Central Tendency
// Mode of the given input array
// Based on
// https://www.mathsisfun.com/data/central-measures.html
pub fn mode[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	mut freqs := []int{cap: t.size}
	mut iter := t.iterator()
	for {
		val, _ := iter.next() or { break }
		freqs << freq(t, val)
	}
	mut max_index := 0
	for i, v in freqs {
		if v > freqs[max_index] {
			max_index = i
		}
	}
	return t.get_nth(max_index)
}

// Root Mean Square of the given input array
// Based on
// https://en.wikipedia.org/wiki/Root_mean_square
pub fn rms[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}
	return math.sqrt(sum_squared_deviations[T](t, vtl.cast[T](0)) / vtl.cast[T](t.size))
}

// Measure of Dispersion / Spread
// Population Variance of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html

// population_variance exposes this operation as part of the public API.

// population_variance exposes this operation as part of the public API.
@[inline]
pub fn population_variance[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	mut t_mean := mean[T](t)
	return population_variance_mean(t, t_mean)
}

// Measure of Dispersion / Spread
// Population Variance of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
pub fn population_variance_mean[T](t &vtl.Tensor[T], provided_mean T) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}
	return sum_squared_deviations[T](t, provided_mean) / vtl.cast[T](t.size)
}

// Measure of Dispersion / Spread
// Sample Variance of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html

// sample_variance exposes this operation as part of the public API.

// sample_variance exposes this operation as part of the public API.
@[inline]
pub fn sample_variance[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	mut t_mean := mean[T](t)
	return sample_variance_mean(t, t_mean)
}

// Measure of Dispersion / Spread
// Sample Variance of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
pub fn sample_variance_mean[T](t &vtl.Tensor[T], provided_mean T) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}
	return sum_squared_deviations[T](t, provided_mean) / vtl.cast[T](t.size - 1)
}

@[direct_array_access]
fn sum_squared_deviations[T](t &vtl.Tensor[T], provided_mean T) T {
	if is_flat_tensor_storage(t) {
		data := t.data.data
		mut total0 := vtl.cast[T](0)
		mut total1 := vtl.cast[T](0)
		mut total2 := vtl.cast[T](0)
		mut total3 := vtl.cast[T](0)
		mut total4 := vtl.cast[T](0)
		mut total5 := vtl.cast[T](0)
		mut total6 := vtl.cast[T](0)
		mut total7 := vtl.cast[T](0)
		mut index := 0
		for index + 7 < data.len {
			difference0 := data[index] - provided_mean
			difference1 := data[index + 1] - provided_mean
			difference2 := data[index + 2] - provided_mean
			difference3 := data[index + 3] - provided_mean
			difference4 := data[index + 4] - provided_mean
			difference5 := data[index + 5] - provided_mean
			difference6 := data[index + 6] - provided_mean
			difference7 := data[index + 7] - provided_mean
			total0 += difference0 * difference0
			total1 += difference1 * difference1
			total2 += difference2 * difference2
			total3 += difference3 * difference3
			total4 += difference4 * difference4
			total5 += difference5 * difference5
			total6 += difference6 * difference6
			total7 += difference7 * difference7
			index += 8
		}
		mut total := (((total0 + total1) + (total2 + total3)) + (total4 + total5)) + (total6 + total7)
		for index < data.len {
			difference := data[index] - provided_mean
			total += difference * difference
			index++
		}
		return total
	}
	return t.reduce(vtl.cast[T](0), fn [provided_mean] [T](acc T, val T, _ []int) T {
		difference := val - provided_mean
		return acc + difference * difference
	})
}

// Measure of Dispersion / Spread
// Population Standard Deviation of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html

// population_stddev exposes this operation as part of the public API.

// population_stddev exposes this operation as part of the public API.
@[inline]
pub fn population_stddev[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	mut t_mean := mean[T](t)
	return population_stddev_mean(t, t_mean)
}

// Measure of Dispersion / Spread
// Population Standard Deviation of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html

// population_stddev_mean exposes this operation as part of the public API.

// population_stddev_mean exposes this operation as part of the public API.
@[inline]
pub fn population_stddev_mean[T](t &vtl.Tensor[T], mean T) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	return math.sqrt(population_variance_mean(t, mean))
}

// Measure of Dispersion / Spread
// Sample Standard Deviation of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html

// sample_stddev exposes this operation as part of the public API.

// sample_stddev exposes this operation as part of the public API.
@[inline]
pub fn sample_stddev[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	mut t_mean := mean[T](t)
	return sample_stddev_mean(t, t_mean)
}

// Measure of Dispersion / Spread
// Sample Standard Deviation of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html

// sample_stddev_mean exposes this operation as part of the public API.

// sample_stddev_mean exposes this operation as part of the public API.
@[inline]
pub fn sample_stddev_mean[T](t &vtl.Tensor[T], mean T) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	return math.sqrt(sample_variance_mean(t, mean))
}

// Measure of Dispersion / Spread
// Mean Absolute Deviation of the given input array
// Based on
// https://en.wikipedia.org/wiki/Average_absolute_deviation

// absdev exposes this operation as part of the public API.

// absdev exposes this operation as part of the public API.
@[inline]
pub fn absdev[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	mut t_mean := mean[T](t)
	return absdev_mean(t, t_mean)
}

// Measure of Dispersion / Spread
// Mean Absolute Deviation of the given input array
// Based on
// https://en.wikipedia.org/wiki/Average_absolute_deviation
pub fn absdev_mean[T](t &vtl.Tensor[T], provided_mean T) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	return t.reduce(vtl.cast[T](0), fn [provided_mean] [T](acc T, val T, i []int) T {
		return acc + math.abs(val - provided_mean)
	}) / vtl.cast[T](t.size)
}

// Sum of squares

// tss exposes this operation as part of the public API.

// tss exposes this operation as part of the public API.
@[inline]
pub fn tss[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	mut t_mean := mean[T](t)
	return tss_mean(t, t_mean)
}

// Sum of squares about the mean
pub fn tss_mean[T](t &vtl.Tensor[T], provided_mean T) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}
	return sum_squared_deviations[T](t, provided_mean)
}

// Minimum of the given input array
pub fn min[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}
	min_value, _ := min_with_index[T](t)
	return min_value
}

// Maximum of the given input array
pub fn max[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	max_value, _ := max_with_index[T](t)
	return max_value
}

// Minimum and maximum of the given input array
pub fn minmax[T](t &vtl.Tensor[T]) (T, T) {
	if t.size == 0 {
		return vtl.cast[T](0), vtl.cast[T](0)
	}

	min_value, max_value, _, _ := minmax_with_indices[T](t)
	return min_value, max_value
}

fn minmax_with_indices[T](t &vtl.Tensor[T]) (T, T, int, int) {
	mut min_value := t.get_nth[T](0)
	mut max_value := min_value
	mut min_at := 0
	mut max_at := 0
	for i in 1 .. t.size {
		value := t.get_nth[T](i)
		if value < min_value {
			min_value = value
			min_at = i
		}
		if value > max_value {
			max_value = value
			max_at = i
		}
	}
	return min_value, max_value, min_at, max_at
}

fn min_with_index[T](t &vtl.Tensor[T]) (T, int) {
	mut min_value := t.get_nth[T](0)
	mut min_at := 0
	for i in 1 .. t.size {
		value := t.get_nth[T](i)
		if value < min_value {
			min_value = value
			min_at = i
		}
	}
	return min_value, min_at
}

fn max_with_index[T](t &vtl.Tensor[T]) (T, int) {
	mut max_value := t.get_nth[T](0)
	mut max_at := 0
	for i in 1 .. t.size {
		value := t.get_nth[T](i)
		if value > max_value {
			max_value = value
			max_at = i
		}
	}
	return max_value, max_at
}

// Minimum of the given input array
pub fn min_index[T](t &vtl.Tensor[T]) int {
	if t.size == 0 {
		return 0
	}

	_, min_at := min_with_index[T](t)
	return min_at
}

// Maximum of the given input array
pub fn max_index[T](t &vtl.Tensor[T]) int {
	if t.size == 0 {
		return 0
	}

	_, max_at := max_with_index[T](t)
	return max_at
}

// Minimum and maximum of the given input array
pub fn minmax_index[T](t &vtl.Tensor[T]) (int, int) {
	if t.size == 0 {
		return 0, 0
	}

	_, _, min_at, max_at := minmax_with_indices[T](t)
	return min_at, max_at
}

// Measure of Dispersion / Spread
// Range ( Maximum - Minimum ) of the given input array
// Based on
// https://www.mathsisfun.com/data/range.html
pub fn range[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	min, max := minmax[T](t)
	return max - min
}

// covariance exposes this operation as part of the public API.

// covariance exposes this operation as part of the public API.
@[inline]
pub fn covariance[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) T {
	mean1 := mean[T](a)
	mean2 := mean[T](b)
	return covariance_mean(a, b, mean1, mean2)
}

// Compute the covariance of a dataset using
// the recurrence relation
pub fn covariance_mean[T](a &vtl.Tensor[T], b &vtl.Tensor[T], mean1 T, mean2 T) T {
	n := math.min(a.size, b.size)
	if n == 0 {
		return vtl.cast[T](0)
	}

	mut cov := vtl.cast[T](0)
	for i in 0 .. n {
		cov += (a[i] - mean1) * (b[i] - mean2)
	}

	return cov / vtl.cast[T](n)
}

// lag1_autocorrelation exposes this operation as part of the public API.

// lag1_autocorrelation exposes this operation as part of the public API.
@[inline]
pub fn lag1_autocorrelation[T](t &vtl.Tensor[T]) T {
	data_mean := mean[T](t)
	return lag1_autocorrelation_mean(t, data_mean)
}

// Compute the lag-1 autocorrelation of a dataset using
// the recurrence relation
pub fn lag1_autocorrelation_mean[T](t &vtl.Tensor[T], provided_mean T) T {
	n := t.size
	if n == 0 {
		return vtl.cast[T](0)
	}

	mut lag1_autocorrelation := vtl.cast[T](0)
	mut lag1_denominator := vtl.cast[T](0)
	for i in 0 .. n - 1 {
		lag1_autocorrelation += (t[i] - provided_mean) * (t[i + 1] - provided_mean)
		lag1_denominator += math.pow(t[i] - provided_mean, vtl.cast[T](2))
	}

	return lag1_autocorrelation / lag1_denominator
}

// kurtosis exposes this operation as part of the public API.

// kurtosis exposes this operation as part of the public API.
@[inline]
pub fn kurtosis[T](t &vtl.Tensor[T]) T {
	data_mean := mean[T](t)
	data_sd := stddev[T](t, data_mean)
	return kurtosis_mean_stddev(t, data_mean, data_sd)
}

// Takes a dataset and finds the kurtosis
// using the fourth moment the deviations, normalized by the sd
pub fn kurtosis_mean_stddev[T](t &vtl.Tensor[T], mean T, sd T) T {
	n := t.size
	if n == 0 {
		return vtl.cast[T](0)
	}

	mut kurtosis := vtl.cast[T](0)
	for i in 0 .. n {
		kurtosis += math.pow(t[i] - mean, vtl.cast[T](4))
	}

	return kurtosis / math.pow(sd, vtl.cast[T](4))
}

// skew exposes this operation as part of the public API.

// skew exposes this operation as part of the public API.
@[inline]
pub fn skew[T](t &vtl.Tensor[T]) T {
	data_mean := mean[T](t)
	data_sd := stddev[T](t, data_mean)
	return skew_mean_stddev(t, data_mean, data_sd)
}

// skew_mean_stddev exposes this operation as part of the public API.
pub fn skew_mean_stddev[T](t &vtl.Tensor[T], mean T, sd T) T {
	n := t.size
	if n == 0 {
		return vtl.cast[T](0)
	}

	mut skew := vtl.cast[T](0)
	for i in 0 .. n {
		skew += math.pow(t[i] - mean, vtl.cast[T](3))
	}

	return skew / math.pow(sd, vtl.cast[T](3))
}

// quantile exposes this operation as part of the public API.
pub fn quantile[T](sorted_t &vtl.Tensor[T], f T) T {
	n := sorted_t.size
	if n == 0 {
		return vtl.cast[T](0)
	}

	index := math.floor(f * vtl.cast[T](n))
	if index == vtl.cast[T](n) {
		index -= vtl.cast[T](1)
	}

	return sorted_t[index]
}

// quantile_linear computes NumPy's default linearly interpolated quantile.
// It sorts a copy of the tensor values and returns an f64, regardless of the
// input element type. NaN input values propagate as NaN.
pub fn quantile_linear[T](t &vtl.Tensor[T], q f64) !f64 {
	if math.is_nan(q) || math.is_inf(q, 0) || q < 0 || q > 1 {
		return error('quantile must be between 0 and 1')
	}
	if t.size == 0 {
		return error('quantile is undefined for an empty tensor')
	}
	mut values := tensor_float64_values[T](t, false)
	return interpolate_quantile(mut values, q)
}

// quantiles_linear computes several NumPy-style quantiles from one sorted
// copy of the flattened tensor values.
pub fn quantiles_linear[T](t &vtl.Tensor[T], quantiles []f64) !&vtl.Tensor[f64] {
	for q in quantiles {
		if math.is_nan(q) || math.is_inf(q, 0) || q < 0 || q > 1 {
			return error('quantiles must be between 0 and 1')
		}
	}
	if t.size == 0 {
		return error('quantiles are undefined for an empty tensor')
	}
	mut values := tensor_float64_values[T](t, false)
	for value in values {
		if math.is_nan(value) {
			return vtl.from_1d[f64]([]f64{len: quantiles.len, init: math.nan()})
		}
	}
	values.sort()
	mut results := []f64{len: quantiles.len}
	for i, q in quantiles {
		results[i] = interpolate_sorted_quantile(values, q)
	}
	return vtl.from_1d[f64](results)
}

// nanquantiles_linear computes several quantiles while ignoring NaN values.
// If every input value is NaN, every result is NaN.
pub fn nanquantiles_linear[T](t &vtl.Tensor[T], quantiles []f64) !&vtl.Tensor[f64] {
	for q in quantiles {
		if math.is_nan(q) || math.is_inf(q, 0) || q < 0 || q > 1 {
			return error('quantiles must be between 0 and 1')
		}
	}
	if t.size == 0 {
		return error('quantiles are undefined for an empty tensor')
	}
	mut values := tensor_float64_values[T](t, true)
	if values.len == 0 {
		return vtl.from_1d[f64]([]f64{len: quantiles.len, init: math.nan()})
	}
	values.sort()
	mut results := []f64{len: quantiles.len}
	for i, q in quantiles {
		results[i] = interpolate_sorted_quantile(values, q)
	}
	return vtl.from_1d[f64](results)
}

// nanquantile_linear computes a linearly interpolated quantile while ignoring
// NaN values. It returns NaN when the tensor has no non-NaN values.
pub fn nanquantile_linear[T](t &vtl.Tensor[T], q f64) !f64 {
	if math.is_nan(q) || math.is_inf(q, 0) || q < 0 || q > 1 {
		return error('quantile must be between 0 and 1')
	}
	if t.size == 0 {
		return error('quantile is undefined for an empty tensor')
	}
	mut values := tensor_float64_values[T](t, true)
	if values.len == 0 {
		return math.nan()
	}
	return interpolate_quantile(mut values, q)
}

// nanmedian returns the median of non-NaN values as f64. Empty and all-NaN
// inputs return NaN.
pub fn nanmedian[T](t &vtl.Tensor[T]) f64 {
	return nanquantile_linear[T](t, 0.5) or { math.nan() }
}

fn interpolate_quantile(mut values []f64, q f64) f64 {
	for value in values {
		if math.is_nan(value) {
			return math.nan()
		}
	}
	position := q * f64(values.len - 1)
	lo := int(math.floor(position))
	hi := math.min(lo + 1, values.len - 1)
	lower := select_order_statistic(mut values, lo)
	if lo == hi {
		return lower
	}
	upper := select_order_statistic(mut values, hi)
	weight := position - f64(lo)
	return lower * (1 - weight) + upper * weight
}

// select_order_statistic partitions values in place and returns its kth value.
// Median-of-three pivots and three-way partitioning handle sorted and
// duplicate-heavy input without sorting the entire buffer.
fn select_order_statistic(mut values []f64, k int) f64 {
	mut low := 0
	mut high := values.len
	for high - low > 1 {
		middle := low + (high - low) / 2
		pivot := median_of_three(values[low], values[middle], values[high - 1])
		mut less := low
		mut current := low
		mut greater := high
		for current < greater {
			if values[current] < pivot {
				values[less], values[current] = values[current], values[less]
				less++
				current++
			} else if values[current] > pivot {
				greater--
				values[current], values[greater] = values[greater], values[current]
			} else {
				current++
			}
		}
		if k < less {
			high = less
		} else if k >= greater {
			low = greater
		} else {
			return values[k]
		}
	}
	return values[k]
}

fn median_of_three(a f64, b f64, c f64) f64 {
	if a > b {
		if b > c {
			return b
		}
		return if a > c { c } else { a }
	}
	if a > c {
		return a
	}
	return if b > c { c } else { b }
}

fn tensor_float64_values[T](t &vtl.Tensor[T], ignore_nan bool) []f64 {
	mut values := []f64{cap: t.size}
	mut iter := t.iterator[T]()
	for {
		value, _ := iter.next() or { break }
		numeric := vtl.cast[f64](value)
		if ignore_nan && math.is_nan(numeric) {
			continue
		}
		values << numeric
	}
	return values
}

fn interpolate_sorted_quantile(values []f64, q f64) f64 {
	position := q * f64(values.len - 1)
	lo := int(math.floor(position))
	hi := math.min(lo + 1, values.len - 1)
	if lo == hi {
		return values[lo]
	}
	weight := position - f64(lo)
	return values[lo] * (1 - weight) + values[hi] * weight
}

// percentile_linear computes NumPy's default linearly interpolated percentile.
// The percentile is expressed on the 0..100 scale.
pub fn percentile_linear[T](t &vtl.Tensor[T], percentile f64) !f64 {
	q := percentile_quantile(percentile)!
	return quantile_linear[T](t, q)
}

// nanpercentile_linear computes a NaN-ignoring linearly interpolated
// percentile on the 0..100 scale.
pub fn nanpercentile_linear[T](t &vtl.Tensor[T], percentile f64) !f64 {
	q := percentile_quantile(percentile)!
	return nanquantile_linear[T](t, q)
}

// percentiles_linear computes several linearly interpolated percentiles on
// the 0..100 scale and returns them in input order.
pub fn percentiles_linear[T](t &vtl.Tensor[T], percentiles []f64) !&vtl.Tensor[f64] {
	return quantiles_linear[T](t, percentiles_to_quantiles(percentiles)!)
}

// nanpercentiles_linear computes several NaN-ignoring percentiles on the
// 0..100 scale and returns them in input order.
pub fn nanpercentiles_linear[T](t &vtl.Tensor[T], percentiles []f64) !&vtl.Tensor[f64] {
	return nanquantiles_linear[T](t, percentiles_to_quantiles(percentiles)!)
}

fn percentile_quantile(percentile f64) !f64 {
	if math.is_nan(percentile) || math.is_inf(percentile, 0) || percentile < 0 || percentile > 100 {
		return error('percentile must be between 0 and 100')
	}
	return percentile / 100
}

fn percentiles_to_quantiles(percentiles []f64) ![]f64 {
	mut quantiles := []f64{len: percentiles.len}
	for i, percentile in percentiles {
		quantiles[i] = percentile_quantile(percentile)!
	}
	return quantiles
}

// percentile_axis computes one linearly interpolated percentile per axis slice
// and retains the reduced axis with length one.
pub fn percentile_axis[T](t &vtl.Tensor[T], percentile f64, axis int) !&vtl.Tensor[f64] {
	q := percentile_quantile(percentile)!
	return quantile_axis[T](t, q, axis)
}

// percentile_axis_squeeze removes the reduced axis from the percentile result.
pub fn percentile_axis_squeeze[T](t &vtl.Tensor[T], percentile f64, axis int) !&vtl.Tensor[f64] {
	q := percentile_quantile(percentile)!
	return quantile_axis_squeeze[T](t, q, axis)
}

// nanpercentile_axis computes a NaN-ignoring percentile per axis slice and
// retains the reduced axis with length one.
pub fn nanpercentile_axis[T](t &vtl.Tensor[T], percentile f64, axis int) !&vtl.Tensor[f64] {
	q := percentile_quantile(percentile)!
	return nanquantile_axis[T](t, q, axis)
}

// nanpercentile_axis_squeeze removes the reduced axis from the NaN-ignoring
// percentile result.
pub fn nanpercentile_axis_squeeze[T](t &vtl.Tensor[T], percentile f64, axis int) !&vtl.Tensor[f64] {
	q := percentile_quantile(percentile)!
	return nanquantile_axis_squeeze[T](t, q, axis)
}

// percentiles_axis computes several percentiles per axis slice. The percentile
// dimension is prepended to the output shape.
pub fn percentiles_axis[T](t &vtl.Tensor[T], percentiles []f64, axis int) !&vtl.Tensor[f64] {
	return quantiles_axis[T](t, percentiles_to_quantiles(percentiles)!, axis)
}

// nanpercentiles_axis computes several NaN-ignoring percentiles per axis slice.
// The percentile dimension is prepended to the output shape.
pub fn nanpercentiles_axis[T](t &vtl.Tensor[T], percentiles []f64, axis int) !&vtl.Tensor[f64] {
	return nanquantiles_axis[T](t, percentiles_to_quantiles(percentiles)!, axis)
}

// quantile_axis computes linearly interpolated quantiles along axis and keeps
// the reduced axis with length one. NaN values propagate within their slice.
pub fn quantile_axis[T](t &vtl.Tensor[T], q f64, axis int) !&vtl.Tensor[f64] {
	return quantile_axis_impl[T](t, q, axis, false, true)
}

// quantile_axis_squeeze computes one quantile per axis slice and removes the
// reduced axis from the result shape, matching NumPy's default quantile shape.
pub fn quantile_axis_squeeze[T](t &vtl.Tensor[T], q f64, axis int) !&vtl.Tensor[f64] {
	return quantile_axis_impl[T](t, q, axis, false, false)
}

// nanquantile_axis computes one linearly interpolated quantile per axis slice,
// ignoring NaN values and retaining the reduced axis with length one. Slices
// containing only NaNs produce NaN.
pub fn nanquantile_axis[T](t &vtl.Tensor[T], q f64, axis int) !&vtl.Tensor[f64] {
	return quantile_axis_impl[T](t, q, axis, true, true)
}

// nanmedian_axis computes the NaN-ignoring median per axis slice and retains
// the reduced axis with length one.
pub fn nanmedian_axis[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nanquantile_axis[T](t, 0.5, axis)
}

// nanquantile_axis_squeeze ignores NaNs and removes the reduced axis from the
// result shape. Slices containing only NaNs produce NaN.
pub fn nanquantile_axis_squeeze[T](t &vtl.Tensor[T], q f64, axis int) !&vtl.Tensor[f64] {
	return quantile_axis_impl[T](t, q, axis, true, false)
}

// nanmedian_axis_squeeze computes the NaN-ignoring median per axis slice and
// removes the reduced axis from the result shape.
pub fn nanmedian_axis_squeeze[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nanquantile_axis_squeeze[T](t, 0.5, axis)
}

// quantiles_axis computes several linearly interpolated quantiles for each
// axis slice. The quantile dimension is prepended to the output shape, matching
// NumPy's quantile(..., axis=axis) layout. Each slice is sorted only once.
// NaN values propagate to every requested quantile in their slice.
pub fn quantiles_axis[T](t &vtl.Tensor[T], quantiles []f64, axis int) !&vtl.Tensor[f64] {
	return quantiles_axis_impl[T](t, quantiles, axis, false)
}

// nanquantiles_axis computes several quantiles per axis slice while ignoring
// NaN values. Slices containing only NaNs produce NaN for each quantile.
pub fn nanquantiles_axis[T](t &vtl.Tensor[T], quantiles []f64, axis int) !&vtl.Tensor[f64] {
	return quantiles_axis_impl[T](t, quantiles, axis, true)
}

fn quantiles_axis_impl[T](t &vtl.Tensor[T], quantiles []f64, axis int, ignore_nan bool) !&vtl.Tensor[f64] {
	for q in quantiles {
		if math.is_nan(q) || math.is_inf(q, 0) || q < 0 || q > 1 {
			return error('quantiles must be between 0 and 1')
		}
	}
	rank := t.rank()
	if rank == 0 {
		return error('quantile axis requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('quantile axis ${axis} out of bounds for rank ${rank}')
	}
	axis_size := t.shape[axis_index]
	if axis_size == 0 {
		return error('quantiles are undefined for an empty axis')
	}
	mut out_shape := [quantiles.len]
	for dim, dimension in t.shape {
		if dim != axis_index {
			out_shape << dimension
		}
	}
	mut result := vtl.empty[f64](out_shape, memory: .row_major)
	mut slice_count := 1
	for dim, dimension in t.shape {
		if dim != axis_index {
			slice_count *= dimension
		}
	}
	mut index := []int{len: rank}
	mut out_index := []int{len: rank}
	mut values := []f64{len: axis_size}
	for slice in 0 .. slice_count {
		decode_nan_slice(slice, t.shape, axis_index, mut index)
		values.clear()
		mut has_nan := false
		for position in 0 .. axis_size {
			index[axis_index] = position
			value := f64(t.get(index))
			if math.is_nan(value) {
				has_nan = true
			} else {
				values << value
			}
		}
		index[axis_index] = 0
		if !ignore_nan && has_nan {
			values.clear()
		} else {
			values.sort()
		}
		out_index[0] = 0
		mut source_dim := 0
		for dim in 0 .. rank {
			if dim != axis_index {
				out_index[source_dim + 1] = index[dim]
				source_dim++
			}
		}
		for quantile_index, q in quantiles {
			out_index[0] = quantile_index
			value := if values.len == 0 {
				math.nan()
			} else {
				interpolate_sorted_quantile(values, q)
			}
			result.set(out_index, value)
		}
	}
	return result
}

fn quantile_axis_impl[T](t &vtl.Tensor[T], q f64, axis int, ignore_nan bool, keepdims bool) !&vtl.Tensor[f64] {
	if math.is_nan(q) || math.is_inf(q, 0) || q < 0 || q > 1 {
		return error('quantile must be between 0 and 1')
	}
	rank := t.rank()
	if rank == 0 {
		return error('quantile axis requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('quantile axis ${axis} out of bounds for rank ${rank}')
	}
	axis_size := t.shape[axis_index]
	if axis_size == 0 {
		return error('quantile is undefined for an empty axis')
	}
	mut out_shape := t.shape.clone()
	if keepdims {
		out_shape[axis_index] = 1
	} else {
		out_shape.delete(axis_index)
	}
	mut result := vtl.empty[f64](out_shape, memory: .row_major)
	mut slice_count := 1
	for dim, dimension in t.shape {
		if dim != axis_index {
			slice_count *= dimension
		}
	}
	mut index := []int{len: rank}
	mut values := []f64{len: axis_size}
	for slice in 0 .. slice_count {
		decode_nan_slice(slice, t.shape, axis_index, mut index)
		values.clear()
		for position in 0 .. axis_size {
			index[axis_index] = position
			value := f64(t.get(index))
			if !ignore_nan || !math.is_nan(value) {
				values << value
			}
		}
		if keepdims {
			index[axis_index] = 0
		}
		output_index := if keepdims {
			index
		} else {
			mut reduced_index := index[..axis_index].clone()
			reduced_index << index[axis_index + 1..]
			reduced_index
		}
		result.set(output_index, if values.len == 0 {
			math.nan()
		} else {
			interpolate_quantile(mut values, q)
		})
	}
	return result
}
