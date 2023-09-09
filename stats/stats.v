module stats

import vtl
import math
import math.stats as math_stats

pub struct AxisData {
	axis int
}

// sum returns the sum of all elements of the given tensor
pub fn sum[T](t &vtl.Tensor[T]) T {
	return t.reduce(vtl.cast[T](0), fn [T](acc T, val T, i []int) T {
		return acc + val
	})
}

// sum_axis returns the sum of a given Tensor along a provided
// axis
pub fn sum_axis[T](t &vtl.Tensor[T], data AxisData) T {
	mut iter := t.axis_iterator(data.axis)
	mut acc := vtl.cast[T](0)
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
	mut acc := vtl.cast[T](0)
	for {
		val, _ := iter.next() or { break }
		acc += val
	}
	return acc
}

// prod returns the product of all elements of the given tensor
pub fn prod[T](t &vtl.Tensor[T]) T {
	/*
	If the tensor is empty, his product is zero
	We are returning it right way otherwise it will be set to 1
	*/
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	return t.reduce(vtl.cast[T](1), fn [T](acc T, val T, i []int) T {
		return acc * val
	})
}

// prod_axis_dims returns the product of a given Tensor along a provided
// axis with the reduced dimension intact
pub fn prod_axis[T](t &vtl.Tensor[T], data AxisData) T {
	mut iter := t.axis_iterator(data.axis)
	mut acc := vtl.cast[T](0)
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
	mut acc := vtl.cast[T](0)
	for {
		val, _ := iter.next() or { break }
		acc *= val
	}
	return acc
}

// Measure of Occurance
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

// Measure of Central Tendancy
// Mean of the given input array
// Based on
// https://www.mathsisfun.com/data/central-measures.html
pub fn mean[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	return t.reduce(vtl.cast[T](0), fn [T](acc T, val T, i []int) T {
		return acc + val
	}) / vtl.cast[T](t.size)
}

// Measure of Central Tendancy
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

// Measure of Central Tendancy
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

// Measure of Central Tendancy
// Median of the given input array ( input array is assumed to be sorted )
// Based on
// https://www.mathsisfun.com/data/central-measures.html
pub fn median[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	if t.size % 2 == 0 {
		return (t.get_nth(t.size / 2) + t.get_nth((t.size / 2) - 1)) / vtl.cast[T](2)
	} else {
		return t.get_nth(t.size / 2)
	}
}

// Measure of Central Tendancy
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

	return math.sqrt(t.reduce(vtl.cast[T](0), fn [T](acc T, val T, i []int) T {
		return acc + math.pow(val, vtl.cast[T](2))
	}) / vtl.cast[T](t.size))
}

// Measure of Dispersion / Spread
// Population Variance of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
[inline]
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

	return t.reduce(vtl.cast[T](0), fn [provided_mean] [T](acc T, val T, i []int) T {
		return acc + math.pow(val - provided_mean, vtl.cast[T](2))
	}) / vtl.cast[T](t.size)
}

// Measure of Dispersion / Spread
// Sample Variance of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
[inline]
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

	return t.reduce(vtl.cast[T](0), fn [provided_mean] [T](acc T, val T, i []int) T {
		return acc + math.pow(val - provided_mean, vtl.cast[T](2))
	}) / vtl.cast[T](t.size - 1)
}

// Measure of Dispersion / Spread
// Population Standard Deviation of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
[inline]
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
[inline]
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
[inline]
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
[inline]
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
[inline]
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
[inline]
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

	return t.reduce(vtl.cast[T](0), fn [provided_mean] [T](acc T, val T, i []int) T {
		return acc + math.pow(val - provided_mean, vtl.cast[T](2))
	})
}

// Minimum of the given input array
pub fn min[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	return math_stats.min[T](t.to_array())
}

// Maximum of the given input array
pub fn max[T](t &vtl.Tensor[T]) T {
	if t.size == 0 {
		return vtl.cast[T](0)
	}

	return math_stats.max[T](t.to_array())
}

// Minimum and maximum of the given input array
pub fn minmax[T](t &vtl.Tensor[T]) (T, T) {
	if t.size == 0 {
		return vtl.cast[T](0), vtl.cast[T](0)
	}

	return math_stats.minmax[T](t.to_array())
}

// Minimum of the given input array
pub fn min_index[T](t &vtl.Tensor[T]) int {
	if t.size == 0 {
		return 0
	}

	return math_stats.min_index[T](t.to_array())
}

// Maximum of the given input array
pub fn max_index[T](t &vtl.Tensor[T]) int {
	if t.size == 0 {
		return 0
	}

	return math_stats.max_index[T](t.to_array())
}

// Minimum and maximum of the given input array
pub fn minmax_index[T](t &vtl.Tensor[T]) (int, int) {
	if t.size == 0 {
		return 0, 0
	}

	return math_stats.minmax_index[T](t.to_array())
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

[inline]
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

[inline]
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

[inline]
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

[inline]
pub fn skew[T](t &vtl.Tensor[T]) T {
	data_mean := mean[T](t)
	data_sd := stddev[T](t, data_mean)
	return skew_mean_stddev(t, data_mean, data_sd)
}

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
