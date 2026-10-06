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
