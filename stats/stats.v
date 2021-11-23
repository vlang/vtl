module stats

import vtl
import math.stats

pub struct AxisData {
	axis int
}

// sum returns the sum of all elements of the given tensor
pub fn sum<T>(t &vtl.Tensor<T>) T {
	mut iter := t.iterator(data.axis)
	mut acc := T(0)
	for {
		val, _ := iter.next() or { break }
		acc += val
	}
	return acc
}

// sum_axis returns the sum of a given Tensor along a provided
// axis
pub fn sum_axis<T>(t &vtl.Tensor<T>, data AxisData) T {
	mut iter := t.axis_iterator(data.axis)
	mut acc := T(0)
	for {
		val, _ := iter.next() or { break }
		acc += val
	}
	return acc
}

// sum_axis_dims returns the sum of a given Tensor along a provided
// axis with the reduced dimension intact
pub fn sum_axis_with_dims<T>(t &vtl.Tensor<T>, data AxisData) T {
	mut iter := t.axis_with_dims_iterator(data.axis)
	mut acc := T(0)
	for {
		val, _ := iter.next() or { break }
		acc += val
	}
	return acc
}

// prod_axis returns the product of a given Tensor along a provided
// axis
pub fn prod<T>(t &vtl.Tensor<T>) T {
	mut iter := t.iterator(data.axis)
	mut acc := T(0)
	for {
		val, _ := iter.next() or { break }
		acc *= val
	}
	return acc
}

// prod_axis_dims returns the product of a given Tensor along a provided
// axis with the reduced dimension intact
pub fn prod_axis<T>(t &vtl.Tensor<T>, data AxisData) T {
	mut iter := t.axis_iterator(data.axis)
	mut acc := T(0)
	for {
		val, _ := iter.next() or { break }
		acc *= val
	}
	return acc
}

// prod_axis_dims returns the product of a Tensor along a provided
// axis with the reduced dimension intact
pub fn prod_axis_with_dims<T>(t &vtl.Tensor<T>, data AxisData) T {
	mut iter := t.axis_with_dims_iterator(data.axis)
	mut acc := T(0)
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
pub fn freq<T>(t &vtl.Tensor<T>, val T) int {
	return stats.freq<T>(t.to_array(), val)
}

// Measure of Central Tendancy
// Mean of the given input array
// Based on
// https://www.mathsisfun.com/data/central-measures.html
pub fn mean<T>(t &vtl.Tensor<T>) T {
	return stats.mean<T>(t.to_array())
}

// Measure of Central Tendancy
// Geometric Mean of the given input array
// Based on
// https://www.mathsisfun.com/numbers/geometric-mean.html
pub fn geometric_mean<T>(t &vtl.Tensor<T>) T {
	return stats.geometric_mean<T>(t.to_array())
}

// Measure of Central Tendancy
// Harmonic Mean of the given input array
// Based on
// https://www.mathsisfun.com/numbers/harmonic-mean.html
pub fn harmonic_mean<T>(t &vtl.Tensor<T>) T {
	return stats.harmonic_mean<T>(t.to_array())
}

// Measure of Central Tendancy
// Median of the given input array ( input array is assumed to be sorted )
// Based on
// https://www.mathsisfun.com/data/central-measures.html
pub fn median<T>(t &vtl.Tensor<T>) T {
	return stats.median<T>(t.to_array())
}

// Measure of Central Tendancy
// Mode of the given input array
// Based on
// https://www.mathsisfun.com/data/central-measures.html
pub fn mode<T>(t &vtl.Tensor<T>) T {
	return stats.mode<T>(t.to_array())
}

// Root Mean Square of the given input array
// Based on
// https://en.wikipedia.org/wiki/Root_mean_square
pub fn rms<T>(t &vtl.Tensor<T>) T {
	return stats.rms<T>(t.to_array())
}

// Measure of Dispersion / Spread
// Population Variance of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
[inline]
pub fn population_variance<T>(t &vtl.Tensor<T>) T {
	return stats.population_variance<T>(t.to_array())
}

// Measure of Dispersion / Spread
// Population Variance of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
pub fn population_variance_mean<T>(t &vtl.Tensor<T>, mean T) T {
	return stats.population_variance_mean<T>(t.to_array(), mean)
}

// Measure of Dispersion / Spread
// Sample Variance of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
[inline]
pub fn sample_variance<T>(t &vtl.Tensor<T>) T {
	return stats.sample_variance<T>(t.to_array())
}

// Measure of Dispersion / Spread
// Sample Variance of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
pub fn sample_variance_mean<T>(t &vtl.Tensor<T>, mean T) T {
	return stats.sample_variance_mean<T>(t.to_array(), mean)
}

// Measure of Dispersion / Spread
// Population Standard Deviation of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
[inline]
pub fn population_stddev<T>(t &vtl.Tensor<T>) T {
	return stats.population_stddev<T>(t.to_array())
}

// Measure of Dispersion / Spread
// Population Standard Deviation of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
[inline]
pub fn population_stddev_mean<T>(t &vtl.Tensor<T>, mean T) T {
	return stats.population_stddev_mean<T>(t.to_array(), mean)
}

// Measure of Dispersion / Spread
// Sample Standard Deviation of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
[inline]
pub fn sample_stddev<T>(t &vtl.Tensor<T>) T {
	return stats.sample_stddev<T>(t.to_array())
}

// Measure of Dispersion / Spread
// Sample Standard Deviation of the given input array
// Based on
// https://www.mathsisfun.com/data/standard-deviation.html
[inline]
pub fn sample_stddev_mean<T>(t &vtl.Tensor<T>, mean T) T {
	return stats.sample_stddev_mean<T>(t.to_array(), mean)
}

// Measure of Dispersion / Spread
// Mean Absolute Deviation of the given input array
// Based on
// https://en.wikipedia.org/wiki/Average_absolute_deviation
[inline]
pub fn absdev<T>(t &vtl.Tensor<T>) T {
	return stats.absdev<T>(t.to_array())
}

// Measure of Dispersion / Spread
// Mean Absolute Deviation of the given input array
// Based on
// https://en.wikipedia.org/wiki/Average_absolute_deviation
pub fn absdev_mean<T>(t &vtl.Tensor<T>, mean T) T {
	return stats.absdev_mean<T>(t.to_array(), mean)
}

// Sum of squares
[inline]
pub fn tss<T>(t &vtl.Tensor<T>) T {
	return stats.tss<T>(t.to_array())
}

// Sum of squares about the mean
pub fn tss_mean<T>(t &vtl.Tensor<T>, mean T) T {
	return stats.tss_mean<T>(t.to_array(), mean)
}

// Minimum of the given input array
pub fn min<T>(t &vtl.Tensor<T>) T {
	return stats.min<T>(t.to_array())
}

// Maximum of the given input array
pub fn max<T>(t &vtl.Tensor<T>) T {
	return stats.max<T>(t.to_array())
}

// Minimum and maximum of the given input array
pub fn minmax<T>(t &vtl.Tensor<T>) (T, T) {
	return stats.minmax<T>(t.to_array())
}

// Minimum of the given input array
pub fn min_index<T>(t &vtl.Tensor<T>) int {
	return stats.min_index<T>(t.to_array())
}

// Maximum of the given input array
pub fn max_index<T>(t &vtl.Tensor<T>) int {
	return stats.max_index<T>(t.to_array())
}

// Minimum and maximum of the given input array
pub fn minmax_index<T>(t &vtl.Tensor<T>) (int, int) {
	return stats.minmax_index<T>(t.to_array())
}

// Measure of Dispersion / Spread
// Range ( Maximum - Minimum ) of the given input array
// Based on
// https://www.mathsisfun.com/data/range.html
pub fn range<T>(t &vtl.Tensor<T>) T {
	return stats.range<T>(t.to_array())
}

[inline]
pub fn covariance<T>(a &vtl.Tensor<T>, b &vtl.Tensor<T>) T {
	return stats.covariance<T>(a.to_array(), b.to_array())
}

// Compute the covariance of a dataset using
// the recurrence relation
pub fn covariance_mean<T>(a &vtl.Tensor<T>, b &vtl.Tensor<T>, mean1 T, mean2 T) T {
	return stats.covariance_mean<T>(a.to_array(), b.to_array(), mean1, mean2)
}

[inline]
pub fn lag1_autocorrelation<T>(t &vtl.Tensor<T>) T {
	return stats.lag1_autocorrelation<T>(t.to_array())
}

// Compute the lag-1 autocorrelation of a dataset using
// the recurrence relation
pub fn lag1_autocorrelation_mean<T>(t &vtl.Tensor<T>, mean T) T {
	return stats.lag1_autocorrelation_mean<T>(t.to_array(), mean)
}

[inline]
pub fn kurtosis<T>(t &vtl.Tensor<T>) T {
	return stats.kurtosis<T>(t.to_array())
}

// Takes a dataset and finds the kurtosis
// using the fourth moment the deviations, normalized by the sd
pub fn kurtosis_mean_stddev<T>(t &vtl.Tensor<T>, mean T, sd T) T {
	return stats.kurtosis_mean_stddev<T>(t.to_array(), mean, sd)
}

[inline]
pub fn skew<T>(t &vtl.Tensor<T>) T {
	return stats.skew<T>(t.to_array())
}

pub fn skew_mean_stddev<T>(t &vtl.Tensor<T>, mean T, sd T) T {
	return stats.skew_mean_stddev<T>(t.to_array(), mean, sd)
}

pub fn quantile<T>(sorted_t &vtl.Tensor<T>, f T) T {
	return stats.quantile<T>(sorted_t.to_array(), f)
}
