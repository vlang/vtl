module stats

import math
import vtl

// Histogram contains bin counts and the corresponding bin edges.
pub struct Histogram {
pub:
	counts    &vtl.Tensor[int]
	bin_edges &vtl.Tensor[f64]
}

// WeightedHistogram contains weighted counts or probability densities and the
// corresponding bin edges.
pub struct WeightedHistogram {
pub:
	counts    &vtl.Tensor[f64]
	bin_edges &vtl.Tensor[f64]
}

// HistogramBinRule selects a standard data-driven bin-width heuristic.
pub enum HistogramBinRule {
	automatic
	doane
	freedman_diaconis
	rice
	scott
	square_root
	stone
	sturges
}

// histogram computes a NumPy-style histogram with evenly spaced bins. The
// range is inferred from finite data; empty input uses [0, 1], and a constant
// input expands its range by 0.5 on each side.
pub fn histogram[T](data &vtl.Tensor[T], bins int) !Histogram {
	if data.size == 0 {
		return histogram_range[T](data, bins, 0, 1)
	}
	mut minimum := math.inf(1)
	mut maximum := math.inf(-1)
	for index in 0 .. data.size {
		value := histogram_value[T](data, index)!
		if math.is_nan(value) || math.is_inf(value, 0) {
			return error('histogram automatic range requires finite input values')
		}
		minimum = math.min(minimum, value)
		maximum = math.max(maximum, value)
	}
	if minimum == maximum {
		minimum -= 0.5
		maximum += 0.5
	}
	return histogram_range[T](data, bins, minimum, maximum)
}

// histogram_auto chooses an evenly spaced bin count from the data. The
// automatic rule uses the larger of Sturges and Freedman-Diaconis estimates,
// falling back to Sturges when the interquartile range is zero.
pub fn histogram_auto[T](data &vtl.Tensor[T], rule HistogramBinRule) !Histogram {
	if data.size == 0 {
		return histogram_range[T](data, 1, 0, 1)
	}
	mut values := []f64{len: data.size}
	mut minimum := math.inf(1)
	mut maximum := math.inf(-1)
	for index in 0 .. data.size {
		value := histogram_value[T](data, index)!
		if math.is_nan(value) || math.is_inf(value, 0) {
			return error('histogram_auto requires finite input values')
		}
		values[index] = value
		minimum = math.min(minimum, value)
		maximum = math.max(maximum, value)
	}
	is_constant := minimum == maximum
	if is_constant {
		minimum -= 0.5
		maximum += 0.5
	}
	sturges_bins := math.max(1, int(math.ceil(math.log2(f64(values.len)))) + 1)
	mut bins := sturges_bins
	match rule {
		.automatic {
			values.sort()
			fd_bins := histogram_freedman_diaconis_bins(values, maximum - minimum)
			bins = math.max(sturges_bins, fd_bins)
		}
		.freedman_diaconis {
			values.sort()
			bins = histogram_freedman_diaconis_bins(values, maximum - minimum)
		}
		.doane {
			bins = histogram_doane_bins(values, sturges_bins)
		}
		.rice {
			bins = math.max(1, int(math.ceil(2 * math.pow(f64(values.len), 1.0 / 3.0))))
		}
		.scott {
			bins = histogram_scott_bins(values, maximum - minimum, sturges_bins)
		}
		.square_root {
			bins = math.max(1, int(math.ceil(math.sqrt(f64(values.len)))))
		}
		.stone {
			bins = if is_constant { 1 } else { histogram_stone_bins(values, minimum, maximum) }
		}
		.sturges {}
	}
	return histogram_range[T](data, bins, minimum, maximum)
}

fn histogram_stone_bins(values []f64, minimum f64, maximum f64) int {
	if values.len <= 1 || minimum == maximum {
		return 1
	}
	n := values.len
	upper_bound := math.max(100, int(math.sqrt(f64(n))))
	mut best_bins := 1
	mut best_score := math.inf(1)
	for candidate in 1 .. upper_bound + 1 {
		mut counts := []int{len: candidate}
		width := (maximum - minimum) / f64(candidate)
		for value in values {
			bin := math.min(int((value - minimum) / width), candidate - 1)
			counts[bin]++
		}
		mut squared_probability_sum := 0.0
		for count in counts {
			probability := f64(count) / f64(n)
			squared_probability_sum += probability * probability
		}
		score := (2 - f64(n + 1) * squared_probability_sum) / width
		if score < best_score {
			best_score = score
			best_bins = candidate
		}
	}
	return best_bins
}

fn histogram_doane_bins(values []f64, fallback int) int {
	if values.len <= 2 {
		return fallback
	}
	mut mean := 0.0
	for value in values {
		mean += value
	}
	mean /= f64(values.len)
	mut second_moment := 0.0
	mut third_moment := 0.0
	for value in values {
		difference := value - mean
		second_moment += difference * difference
		third_moment += difference * difference * difference
	}
	second_moment /= f64(values.len)
	if second_moment == 0 {
		return fallback
	}
	third_moment /= f64(values.len)
	skewness := third_moment / math.pow(second_moment, 1.5)
	n := f64(values.len)
	skewness_error := math.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
	if skewness_error == 0 {
		return fallback
	}
	bin_count := 1 + math.log2(n) + math.log2(1 + math.abs(skewness) / skewness_error)
	return math.max(1, int(math.ceil(bin_count)))
}

fn histogram_freedman_diaconis_bins(sorted []f64, data_range f64) int {
	if sorted.len < 2 {
		return 1
	}
	interquartile_range := histogram_sample_quantile(sorted, 0.75) - histogram_sample_quantile(sorted,
		0.25)
	if interquartile_range <= 0 {
		return math.max(1, int(math.ceil(math.log2(f64(sorted.len)))) + 1)
	}
	width := 2 * interquartile_range / math.pow(f64(sorted.len), 1.0 / 3.0)
	return math.max(1, int(math.ceil(data_range / width)))
}

fn histogram_scott_bins(values []f64, data_range f64, fallback int) int {
	if values.len < 2 {
		return 1
	}
	mut mean := 0.0
	for value in values {
		mean += value
	}
	mean /= f64(values.len)
	mut sum_squared_deviations := 0.0
	for value in values {
		difference := value - mean
		sum_squared_deviations += difference * difference
	}
	standard_deviation := math.sqrt(sum_squared_deviations / f64(values.len))
	if standard_deviation == 0 {
		return fallback
	}
	width := 3.5 * standard_deviation / math.pow(f64(values.len), 1.0 / 3.0)
	return math.max(1, int(math.ceil(data_range / width)))
}

fn histogram_sample_quantile(sorted []f64, q f64) f64 {
	position := q * f64(sorted.len - 1)
	lower := int(math.floor(position))
	upper := math.min(lower + 1, sorted.len - 1)
	weight := position - f64(lower)
	return sorted[lower] * (1 - weight) + sorted[upper] * weight
}

// histogram_range computes evenly spaced bins over [minimum, maximum]. Values
// outside the range are ignored; the final bin includes its right edge.
pub fn histogram_range[T](data &vtl.Tensor[T], bins int, minimum f64, maximum f64) !Histogram {
	if bins <= 0 {
		return error('histogram requires a positive bin count')
	}
	if math.is_nan(minimum) || math.is_inf(minimum, 0) || math.is_nan(maximum)
		|| math.is_inf(maximum, 0) || minimum >= maximum {
		return error('histogram range must be finite and increasing')
	}
	mut edges := []f64{len: bins + 1}
	width := (maximum - minimum) / f64(bins)
	for index in 0 .. bins {
		edges[index] = minimum + f64(index) * width
	}
	edges[bins] = maximum
	mut counts := []int{len: bins}
	for index in 0 .. data.size {
		value := histogram_value[T](data, index)!
		if math.is_nan(value) || value < minimum || value > maximum {
			continue
		}
		bin := histogram_bin(value, edges, bins)
		counts[bin]++
	}
	return Histogram{
		counts:    vtl.from_1d(counts)!
		bin_edges: vtl.from_1d(edges)!
	}
}

// histogram_edges computes unweighted counts using explicit, strictly
// increasing bin edges. The final bin includes its right edge.
pub fn histogram_edges[T](data &vtl.Tensor[T], edges []f64) !Histogram {
	validate_histogram_edges(edges)!
	bins := edges.len - 1
	mut counts := []int{len: bins}
	for index in 0 .. data.size {
		value := histogram_value[T](data, index)!
		if math.is_nan(value) || value < edges[0] || value > edges[bins] {
			continue
		}
		counts[histogram_bin(value, edges, bins)]++
	}
	return Histogram{
		counts:    vtl.from_1d(counts)!
		bin_edges: vtl.from_1d(edges)!
	}
}

// histogram_weighted_edges accumulates one numeric weight per input value
// using explicit bin edges. With density enabled, the weighted result is
// normalized so the sum of density times bin width is one.
pub fn histogram_weighted_edges[T, W](data &vtl.Tensor[T], weights &vtl.Tensor[W], edges []f64, density bool) !WeightedHistogram {
	validate_histogram_edges(edges)!
	if data.shape != weights.shape {
		return error('histogram weights must have the same shape as data')
	}
	bins := edges.len - 1
	mut counts := []f64{len: bins}
	for index in 0 .. data.size {
		value := histogram_value[T](data, index)!
		if math.is_nan(value) || value < edges[0] || value > edges[bins] {
			continue
		}
		weight := histogram_value[W](weights, index)!
		if math.is_nan(weight) || math.is_inf(weight, 0) {
			return error('histogram weights must be finite')
		}
		counts[histogram_bin(value, edges, bins)] += weight
	}
	if density {
		mut total_weight := 0.0
		for count in counts {
			total_weight += count
		}
		if total_weight == 0 || math.is_nan(total_weight) || math.is_inf(total_weight, 0) {
			return error('histogram density requires a finite nonzero total weight')
		}
		for index in 0 .. bins {
			width := edges[index + 1] - edges[index]
			counts[index] /= total_weight * width
		}
	}
	return WeightedHistogram{
		counts:    vtl.from_1d(counts)!
		bin_edges: vtl.from_1d(edges)!
	}
}

fn validate_histogram_edges(edges []f64) ! {
	if edges.len < 2 {
		return error('histogram requires at least two bin edges')
	}
	for index, edge in edges {
		if math.is_nan(edge) || math.is_inf(edge, 0) {
			return error('histogram bin edges must be finite')
		}
		if index > 0 && edge <= edges[index - 1] {
			return error('histogram bin edges must be strictly increasing')
		}
	}
}

fn histogram_bin(value f64, edges []f64, bins int) int {
	if value == edges[bins] {
		return bins - 1
	}
	mut low := 0
	mut high := bins
	for low < high {
		middle := low + (high - low) / 2
		if value >= edges[middle + 1] {
			low = middle + 1
		} else {
			high = middle
		}
	}
	return low
}

fn histogram_value[T](data &vtl.Tensor[T], index int) !f64 {
	$if T is bool || T is string {
		return error('histogram requires a numeric tensor')
	} $else {
		return f64(data.get_nth(index))
	}
}
