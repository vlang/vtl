module stats

import math
import vtl

// HistogramDD contains counts for N-dimensional sample points and one edge
// vector per sample dimension. Counts use row-major bin order.
pub struct HistogramDD {
pub:
	counts &vtl.Tensor[int]
	edges  [][]f64
}

// WeightedHistogramDD contains weighted counts or probability densities.
pub struct WeightedHistogramDD {
pub:
	counts &vtl.Tensor[f64]
	edges  [][]f64
}

// histogramdd computes a NumPy-style N-dimensional histogram with one
// uniformly spaced bin count per sample dimension. Empty dimensions use [0, 1]
// as their inferred range; constant dimensions expand by 0.5 on either side.
pub fn histogramdd[T](sample &vtl.Tensor[T], bins []int) !HistogramDD {
	edges := histogramdd_inferred_edges[T](sample, bins)!
	return histogramdd_edges[T](sample, edges)
}

// histogramdd_edges counts sample rows using one strictly increasing edge
// vector per dimension. Points outside any edge range or containing NaN are
// ignored; each dimension's final bin includes its right edge.
pub fn histogramdd_edges[T](sample &vtl.Tensor[T], edges [][]f64) !HistogramDD {
	validate_histogramdd_sample(sample)!
	dimensions := sample.shape[1]
	if edges.len != dimensions {
		return error('histogramdd requires one edge vector per sample dimension')
	}
	bins := histogramdd_validate_edges(edges, dimensions)!
	counts_len := histogramdd_bin_count(bins)!
	widths := histogramdd_edge_widths(edges)
	mut counts := []int{len: counts_len}
	mut bin_indices := []int{len: dimensions}
	for row in 0 .. sample.shape[0] {
		mut included := true
		for dimension in 0 .. dimensions {
			value := histogram_value[T](sample, row * dimensions + dimension)!
			axis_edges := edges[dimension]
			if math.is_nan(value) || value < axis_edges[0] || value > axis_edges[axis_edges.len - 1] {
				included = false
				break
			}
			bin_indices[dimension] = histogram_bin_with_uniform_width(value, axis_edges,
				bins[dimension],
				widths[dimension])
		}
		if included {
			flat_index := histogramdd_flat_index(bin_indices, bins)
			counts[flat_index]++
		}
	}
	return HistogramDD{
		counts: vtl.from_array[int](counts, bins)!
		edges:  histogramdd_clone_edges(edges)
	}
}

// histogramdd_density computes the probability density for inferred uniform
// bins. Density integrates to one across all included samples.
pub fn histogramdd_density[T](sample &vtl.Tensor[T], bins []int) !WeightedHistogramDD {
	histogram := histogramdd[T](sample, bins)!
	return histogramdd_density_from_counts(histogram)
}

// histogramdd_density_edges computes a probability density using explicit
// edges. Points outside the edge ranges are excluded from normalization.
pub fn histogramdd_density_edges[T](sample &vtl.Tensor[T], edges [][]f64) !WeightedHistogramDD {
	histogram := histogramdd_edges[T](sample, edges)!
	return histogramdd_density_from_counts(histogram)
}

// histogramdd_weighted computes weighted counts or density with inferred
// uniform edges. Density integrates to one over the N-dimensional bin volumes.
pub fn histogramdd_weighted[T, W](sample &vtl.Tensor[T], weights &vtl.Tensor[W], bins []int, density bool) !WeightedHistogramDD {
	validate_histogramdd_weights(sample, weights)!
	edges := histogramdd_inferred_edges[T](sample, bins)!
	return histogramdd_weighted_edges[T, W](sample, weights, edges, density)
}

// histogramdd_weighted_edges accumulates one finite weight per sample row.
// When density is true, each count is divided by total weight and bin volume.
pub fn histogramdd_weighted_edges[T, W](sample &vtl.Tensor[T], weights &vtl.Tensor[W], edges [][]f64, density bool) !WeightedHistogramDD {
	validate_histogramdd_sample(sample)!
	validate_histogramdd_weights(sample, weights)!
	dimensions := sample.shape[1]
	bins := histogramdd_validate_edges(edges, dimensions)!
	counts_len := histogramdd_bin_count(bins)!
	widths := histogramdd_edge_widths(edges)
	mut counts := []f64{len: counts_len}
	mut bin_indices := []int{len: dimensions}
	for row in 0 .. sample.shape[0] {
		mut included := true
		for dimension in 0 .. dimensions {
			value := histogram_value[T](sample, row * dimensions + dimension)!
			axis_edges := edges[dimension]
			if math.is_nan(value) || value < axis_edges[0] || value > axis_edges[axis_edges.len - 1] {
				included = false
				break
			}
			bin_indices[dimension] = histogram_bin_with_uniform_width(value, axis_edges,
				bins[dimension],
				widths[dimension])
		}
		if included {
			weight := histogram_value[W](weights, row)!
			if math.is_nan(weight) || math.is_inf(weight, 0) {
				return error('histogramdd weights must be finite')
			}
			counts[histogramdd_flat_index(bin_indices, bins)] += weight
		}
	}
	if density {
		mut total_weight := 0.0
		for count in counts {
			total_weight += count
		}
		if total_weight == 0 || math.is_nan(total_weight) || math.is_inf(total_weight, 0) {
			return error('histogramdd density requires a finite nonzero total weight')
		}
		for flat_index in 0 .. counts.len {
			volume := histogramdd_bin_volume(flat_index, bins, edges)
			if volume <= 0 || math.is_nan(volume) || math.is_inf(volume, 0) {
				return error('histogramdd bin volume must be finite and positive')
			}
			normalizer := total_weight * volume
			if normalizer == 0 || math.is_nan(normalizer) || math.is_inf(normalizer, 0) {
				return error('histogramdd density normalization must be finite and nonzero')
			}
			counts[flat_index] /= normalizer
		}
	}
	return WeightedHistogramDD{
		counts: vtl.from_array[f64](counts, bins)!
		edges:  histogramdd_clone_edges(edges)
	}
}

fn validate_histogramdd_sample[T](sample &vtl.Tensor[T]) ! {
	if sample.rank() != 2 || sample.shape[1] <= 0 {
		return error('histogramdd expects a [samples, dimensions] tensor with at least one dimension')
	}
}

fn validate_histogramdd_weights[T, W](sample &vtl.Tensor[T], weights &vtl.Tensor[W]) ! {
	validate_histogramdd_sample[T](sample)!
	if weights.rank() != 1 || weights.size != sample.shape[0] {
		return error('histogramdd weights must contain one value per sample row')
	}
}

fn histogramdd_inferred_edges[T](sample &vtl.Tensor[T], bins []int) ![][]f64 {
	validate_histogramdd_sample(sample)!
	dimensions := sample.shape[1]
	if bins.len != dimensions {
		return error('histogramdd requires one bin count per sample dimension')
	}
	_ := histogramdd_bin_count(bins)!
	mut edges := [][]f64{len: dimensions}
	for dimension in 0 .. dimensions {
		mut minimum := math.inf(1)
		mut maximum := math.inf(-1)
		if sample.shape[0] == 0 {
			minimum = 0
			maximum = 1
		} else {
			for row in 0 .. sample.shape[0] {
				value := histogram_value[T](sample, row * dimensions + dimension)!
				if math.is_nan(value) || math.is_inf(value, 0) {
					return error('histogramdd automatic range requires finite input values')
				}
				minimum = math.min(minimum, value)
				maximum = math.max(maximum, value)
			}
			if minimum == maximum {
				minimum -= 0.5
				maximum += 0.5
			}
		}
		edges[dimension] = uniform_histogram_edges(bins[dimension], minimum, maximum)!
	}
	return edges
}

fn histogramdd_validate_edges(edges [][]f64, dimensions int) ![]int {
	if edges.len != dimensions {
		return error('histogramdd requires one edge vector per sample dimension')
	}
	mut bins := []int{len: dimensions}
	for dimension in 0 .. dimensions {
		validate_histogram_edges(edges[dimension])!
		bins[dimension] = edges[dimension].len - 1
	}
	_ := histogramdd_bin_count(bins)!
	return bins
}

fn histogramdd_bin_count(bins []int) !int {
	mut count := 1
	for axis_bins in bins {
		if axis_bins <= 0 || axis_bins == max_int || count > max_int / axis_bins {
			return error('histogramdd bin shape overflows')
		}
		count *= axis_bins
	}
	return count
}

fn histogramdd_flat_index(indices []int, bins []int) int {
	mut flat_index := 0
	for dimension in 0 .. bins.len {
		flat_index = flat_index * bins[dimension] + indices[dimension]
	}
	return flat_index
}

fn histogramdd_clone_edges(edges [][]f64) [][]f64 {
	mut result := [][]f64{len: edges.len}
	for dimension, axis_edges in edges {
		result[dimension] = axis_edges.clone()
	}
	return result
}

fn histogramdd_edge_widths(edges [][]f64) []f64 {
	mut widths := []f64{len: edges.len}
	for dimension, axis_edges in edges {
		widths[dimension] = uniform_histogram_width(axis_edges)
	}
	return widths
}

fn histogramdd_bin_volume(flat_index int, bins []int, edges [][]f64) f64 {
	mut remainder := flat_index
	mut volume := 1.0
	for dimension := bins.len - 1; dimension >= 0; dimension-- {
		bin_index := remainder % bins[dimension]
		remainder /= bins[dimension]
		axis_edges := edges[dimension]
		volume *= axis_edges[bin_index + 1] - axis_edges[bin_index]
	}
	return volume
}

fn histogramdd_density_from_counts(histogram HistogramDD) !WeightedHistogramDD {
	counts := histogram.counts.data.data
	mut total_count := 0.0
	for count in counts[..histogram.counts.size] {
		total_count += f64(count)
	}
	if total_count == 0 || math.is_nan(total_count) || math.is_inf(total_count, 0) {
		return error('histogramdd density requires a finite nonzero sample count')
	}
	bins := histogram.counts.shape
	mut density := []f64{len: histogram.counts.size}
	for flat_index in 0 .. density.len {
		volume := histogramdd_bin_volume(flat_index, bins, histogram.edges)
		if volume <= 0 || math.is_nan(volume) || math.is_inf(volume, 0) {
			return error('histogramdd bin volume must be finite and positive')
		}
		normalizer := total_count * volume
		if normalizer == 0 || math.is_nan(normalizer) || math.is_inf(normalizer, 0) {
			return error('histogramdd density normalization must be finite and nonzero')
		}
		density[flat_index] = f64(counts[flat_index]) / normalizer
	}
	return WeightedHistogramDD{
		counts: vtl.from_array[f64](density, bins)!
		edges:  histogram.edges
	}
}
