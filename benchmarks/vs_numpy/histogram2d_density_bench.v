// Compare current VTL density code, its previous copy-based path, and NumPy.
// Compile with -prod from ~/.vmodules, then execute the binary separately.
module main

import math
import time
import vtl
import vtl.stats

const sample_count = 262_144
const bin_count = 1024
const repetitions = 7

fn main() {
	mut x_values := []f64{len: sample_count}
	mut y_values := []f64{len: sample_count}
	for index in 0 .. sample_count {
		x_values[index] = f64((index * 17) % 65_536) / 65_535.0
		y_values[index] = f64((index * 29 + 7) % 65_536) / 65_535.0
	}
	x := vtl.from_1d(x_values)!
	y := vtl.from_1d(y_values)!
	mut edges := []f64{len: bin_count + 1}
	for index in 0 .. bin_count + 1 {
		edges[index] = f64(index) / f64(bin_count)
	}
	for _ in 0 .. 2 {
		_ := stats.histogram2d_density_edges[f64, f64](x, y, edges, edges)!
		_ := density_copy_baseline(x, y, edges)!
	}
	mut current_times := []f64{len: repetitions}
	mut copy_times := []f64{len: repetitions}
	mut current_checksum := 0.0
	mut copy_checksum := 0.0
	for repetition in 0 .. repetitions {
		if repetition % 2 == 0 {
			started := time.sys_mono_now()
			current := stats.histogram2d_density_edges[f64, f64](x, y, edges, edges)!
			current_times[repetition] = f64(time.sys_mono_now() - started) / 1_000_000.0
			current_checksum += density_checksum(current.counts)

			copy_started := time.sys_mono_now()
			copied := density_copy_baseline(x, y, edges)!
			copy_times[repetition] = f64(time.sys_mono_now() - copy_started) / 1_000_000.0
			copy_checksum += density_checksum(copied.counts)
		} else {
			started := time.sys_mono_now()
			copied := density_copy_baseline(x, y, edges)!
			copy_times[repetition] = f64(time.sys_mono_now() - started) / 1_000_000.0
			copy_checksum += density_checksum(copied.counts)

			current_started := time.sys_mono_now()
			current := stats.histogram2d_density_edges[f64, f64](x, y, edges, edges)!
			current_times[repetition] = f64(time.sys_mono_now() - current_started) / 1_000_000.0
			current_checksum += density_checksum(current.counts)
		}
	}
	current_times.sort()
	copy_times.sort()
	println('operation=histogram2d_density samples=262144 bins=1024x1024 repetitions=7')
	println('current_median_ms=' + current_times[repetitions / 2].str())
	println('previous_copy_path_median_ms=' + copy_times[repetitions / 2].str())
	println('current_weighted_checksum=' + (current_checksum / f64(repetitions)).str())
	println('previous_weighted_checksum=' + (copy_checksum / f64(repetitions)).str())
}

@[direct_array_access]
fn density_checksum(counts &vtl.Tensor[f64]) f64 {
	mut checksum := 0.0
	for index in 0 .. counts.size {
		checksum += counts.data.data[index] * f64(index + 1)
	}
	return checksum
}

// density_copy_baseline reproduces the previous implementation's array copies.
fn density_copy_baseline(x &vtl.Tensor[f64], y &vtl.Tensor[f64], edges []f64) !stats.WeightedHistogram2D {
	histogram := stats.histogram2d_edges[f64, f64](x, y, edges, edges)!
	counts := histogram.counts.to_array().map(f64(it))
	x_edges := histogram.x_edges.to_array()
	y_edges := histogram.y_edges.to_array()
	mut total_count := 0.0
	for count in counts {
		total_count += count
	}
	if total_count == 0 || math.is_nan(total_count) || math.is_inf(total_count, 0) {
		return error('histogram2d density requires a finite nonzero sample count')
	}
	mut density := counts.clone()
	x_bins := x_edges.len - 1
	y_bins := y_edges.len - 1
	for x_bin in 0 .. x_bins {
		for y_bin in 0 .. y_bins {
			index := x_bin * y_bins + y_bin
			area := (x_edges[x_bin + 1] - x_edges[x_bin]) * (y_edges[y_bin + 1] - y_edges[y_bin])
			density[index] /= total_count * area
		}
	}
	return stats.WeightedHistogram2D{
		counts:  vtl.from_array[f64](density, [x_bins, y_bins])!
		x_edges: vtl.from_1d(x_edges)!
		y_edges: vtl.from_1d(y_edges)!
	}
}
