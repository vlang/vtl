module stats

import math
import vtl

// Histogram2D contains a two-dimensional count grid and its bin edges.
// counts is indexed as [x_bin, y_bin], matching NumPy histogram2d.
pub struct Histogram2D {
pub:
	counts  &vtl.Tensor[int]
	x_edges &vtl.Tensor[f64]
	y_edges &vtl.Tensor[f64]
}

// WeightedHistogram2D contains weighted counts or probability densities.
pub struct WeightedHistogram2D {
pub:
	counts  &vtl.Tensor[f64]
	x_edges &vtl.Tensor[f64]
	y_edges &vtl.Tensor[f64]
}

// histogram2d computes a NumPy-style two-dimensional histogram. Inputs are
// flattened in logical order. Empty inputs use [0, 1] on both axes; constant
// inputs expand by 0.5 on either side.
pub fn histogram2d[X, Y](x &vtl.Tensor[X], y &vtl.Tensor[Y], bins_x int, bins_y int) !Histogram2D {
	validate_histogram2d_inputs[X, Y](x, y)!
	if bins_x <= 0 || bins_y <= 0 {
		return error('histogram2d requires positive bin counts')
	}
	x_edges := histogram2d_inferred_edges[X](x, bins_x)!
	y_edges := histogram2d_inferred_edges[Y](y, bins_y)!
	return histogram2d_edges[X, Y](x, y, x_edges, y_edges)
}

// histogram2d_edges counts paired values using strictly increasing bin edges.
// Values outside either edge range are ignored; the last bin includes its
// right edge. The returned count tensor has shape [x_bins, y_bins].
pub fn histogram2d_edges[X, Y](x &vtl.Tensor[X], y &vtl.Tensor[Y], x_edges []f64, y_edges []f64) !Histogram2D {
	validate_histogram2d_inputs[X, Y](x, y)!
	validate_histogram_edges(x_edges)!
	validate_histogram_edges(y_edges)!
	x_bins := x_edges.len - 1
	y_bins := y_edges.len - 1
	x_width := uniform_histogram_width(x_edges)
	y_width := uniform_histogram_width(y_edges)
	mut counts := []int{len: x_bins * y_bins}
	for index in 0 .. x.size {
		x_value := histogram_value[X](x, index)!
		y_value := histogram_value[Y](y, index)!
		if math.is_nan(x_value) || math.is_nan(y_value) || x_value < x_edges[0]
			|| x_value > x_edges[x_bins] || y_value < y_edges[0]
			|| y_value > y_edges[y_bins] {
			continue
		}
		x_bin := histogram2d_find_bin(x_value, x_edges, x_bins, x_width)
		y_bin := histogram2d_find_bin(y_value, y_edges, y_bins, y_width)
		counts[x_bin * y_bins + y_bin]++
	}
	return Histogram2D{
		counts:  vtl.from_array[int](counts, [x_bins, y_bins])!
		x_edges: vtl.from_1d(x_edges)!
		y_edges: vtl.from_1d(y_edges)!
	}
}

// histogram2d_density returns a probability density whose integral over the
// returned bin areas is one. Samples outside the inferred range are excluded.
pub fn histogram2d_density[X, Y](x &vtl.Tensor[X], y &vtl.Tensor[Y], bins_x int, bins_y int) !WeightedHistogram2D {
	histogram := histogram2d[X, Y](x, y, bins_x, bins_y)!
	return histogram2d_density_from_counts(histogram)
}

// histogram2d_density_edges computes a density over explicit bin edges.
pub fn histogram2d_density_edges[X, Y](x &vtl.Tensor[X], y &vtl.Tensor[Y], x_edges []f64, y_edges []f64) !WeightedHistogram2D {
	histogram := histogram2d_edges[X, Y](x, y, x_edges, y_edges)!
	return histogram2d_density_from_counts(histogram)
}

// histogram2d_weighted computes weighted counts or a probability density using
// uniformly spaced bins inferred from the paired input values.
pub fn histogram2d_weighted[X, Y, W](x &vtl.Tensor[X], y &vtl.Tensor[Y], weights &vtl.Tensor[W], bins_x int, bins_y int, density bool) !WeightedHistogram2D {
	validate_histogram2d_inputs[X, Y](x, y)!
	if weights.rank() != 1 || weights.size != x.size {
		return error('histogram2d weights must have the same number of values as x and y')
	}
	if bins_x <= 0 || bins_y <= 0 {
		return error('histogram2d requires positive bin counts')
	}
	x_edges := histogram2d_inferred_edges[X](x, bins_x)!
	y_edges := histogram2d_inferred_edges[Y](y, bins_y)!
	return histogram2d_weighted_edges[X, Y, W](x, y, weights, x_edges, y_edges, density)
}

// histogram2d_weighted_edges accumulates one numeric weight per paired value.
// With density enabled, the sum of density times each bin's area is one.
pub fn histogram2d_weighted_edges[X, Y, W](x &vtl.Tensor[X], y &vtl.Tensor[Y], weights &vtl.Tensor[W], x_edges []f64, y_edges []f64, density bool) !WeightedHistogram2D {
	validate_histogram2d_inputs[X, Y](x, y)!
	if weights.rank() != 1 || weights.size != x.size {
		return error('histogram2d weights must have the same number of values as x and y')
	}
	validate_histogram_edges(x_edges)!
	validate_histogram_edges(y_edges)!
	x_bins := x_edges.len - 1
	y_bins := y_edges.len - 1
	x_width := uniform_histogram_width(x_edges)
	y_width := uniform_histogram_width(y_edges)
	mut counts := []f64{len: x_bins * y_bins}
	for index in 0 .. x.size {
		x_value := histogram_value[X](x, index)!
		y_value := histogram_value[Y](y, index)!
		if math.is_nan(x_value) || math.is_nan(y_value) || x_value < x_edges[0]
			|| x_value > x_edges[x_bins] || y_value < y_edges[0]
			|| y_value > y_edges[y_bins] {
			continue
		}
		weight := histogram_value[W](weights, index)!
		if math.is_nan(weight) || math.is_inf(weight, 0) {
			return error('histogram2d weights must be finite')
		}
		x_bin := histogram2d_find_bin(x_value, x_edges, x_bins, x_width)
		y_bin := histogram2d_find_bin(y_value, y_edges, y_bins, y_width)
		counts[x_bin * y_bins + y_bin] += weight
	}
	if density {
		mut total_weight := 0.0
		for count in counts {
			total_weight += count
		}
		if total_weight == 0 || math.is_nan(total_weight) || math.is_inf(total_weight, 0) {
			return error('histogram2d density requires a finite nonzero total weight')
		}
		counts = normalize_histogram2d_density(counts, x_edges, y_edges, total_weight)!
	}
	return WeightedHistogram2D{
		counts:  vtl.from_array[f64](counts, [x_bins, y_bins])!
		x_edges: vtl.from_1d(x_edges)!
		y_edges: vtl.from_1d(y_edges)!
	}
}

fn histogram2d_find_bin(value f64, edges []f64, bins int, uniform_width f64) int {
	if uniform_width <= 0 {
		return histogram_bin(value, edges, bins)
	}
	if value == edges[bins] {
		return bins - 1
	}
	mut bin := int((value - edges[0]) / uniform_width)
	bin = math.max(0, math.min(bin, bins - 1))
	for bin > 0 && value < edges[bin] {
		bin--
	}
	for bin < bins - 1 && value >= edges[bin + 1] {
		bin++
	}
	return bin
}

fn uniform_histogram_width(edges []f64) f64 {
	bins := edges.len - 1
	width := (edges[bins] - edges[0]) / f64(bins)
	for index in 1 .. bins {
		if edges[index] != edges[0] + f64(index) * width {
			return 0
		}
	}
	return width
}

fn validate_histogram2d_inputs[X, Y](x &vtl.Tensor[X], y &vtl.Tensor[Y]) ! {
	if x.rank() != 1 || y.rank() != 1 {
		return error('histogram2d expects one-dimensional x and y inputs')
	}
	if x.size != y.size {
		return error('histogram2d x and y must contain the same number of values')
	}
}

fn histogram2d_inferred_edges[T](data &vtl.Tensor[T], bins int) ![]f64 {
	if data.size == 0 {
		return uniform_histogram_edges(bins, 0, 1)
	}
	mut minimum := math.inf(1)
	mut maximum := math.inf(-1)
	for index in 0 .. data.size {
		value := histogram_value[T](data, index)!
		if math.is_nan(value) || math.is_inf(value, 0) {
			return error('histogram2d automatic range requires finite input values')
		}
		minimum = math.min(minimum, value)
		maximum = math.max(maximum, value)
	}
	if minimum == maximum {
		minimum -= 0.5
		maximum += 0.5
	}
	return uniform_histogram_edges(bins, minimum, maximum)
}

fn uniform_histogram_edges(bins int, minimum f64, maximum f64) ![]f64 {
	if bins <= 0 {
		return error('histogram2d requires a positive bin count')
	}
	mut edges := []f64{len: bins + 1}
	width := (maximum - minimum) / f64(bins)
	for index in 0 .. bins {
		edges[index] = minimum + f64(index) * width
	}
	edges[bins] = maximum
	return edges
}

@[direct_array_access]
fn histogram2d_density_from_counts(histogram Histogram2D) !WeightedHistogram2D {
	counts := histogram.counts.data.data
	x_edges := histogram.x_edges.data.data
	y_edges := histogram.y_edges.data.data
	x_bins := histogram.counts.shape[0]
	y_bins := histogram.counts.shape[1]
	mut total_count := 0.0
	for count in counts[..histogram.counts.size] {
		total_count += f64(count)
	}
	if total_count == 0 || math.is_nan(total_count) || math.is_inf(total_count, 0) {
		return error('histogram2d density requires a finite nonzero sample count')
	}
	mut density := []f64{len: histogram.counts.size}
	for x_bin in 0 .. x_bins {
		for y_bin in 0 .. y_bins {
			index := x_bin * y_bins + y_bin
			area := (x_edges[x_bin + 1] - x_edges[x_bin]) * (y_edges[y_bin + 1] - y_edges[y_bin])
			density[index] = f64(counts[index]) / (total_count * area)
		}
	}
	return WeightedHistogram2D{
		counts:  vtl.from_array[f64](density, [x_bins, y_bins])!
		x_edges: histogram.x_edges
		y_edges: histogram.y_edges
	}
}

fn normalize_histogram2d_density(counts []f64, x_edges []f64, y_edges []f64, total f64) ![]f64 {
	x_bins := x_edges.len - 1
	y_bins := y_edges.len - 1
	mut density := counts.clone()
	for x_bin in 0 .. x_bins {
		for y_bin in 0 .. y_bins {
			area := (x_edges[x_bin + 1] - x_edges[x_bin]) * (y_edges[y_bin + 1] - y_edges[y_bin])
			density[x_bin * y_bins + y_bin] /= total * area
		}
	}
	return density
}
