// Compare VTL's explicit-edge 2D histogram density with NumPy on identical samples.
// Compile with -prod from ~/.vmodules, then execute the binary separately.
module main

import time
import vtl
import vtl.stats

const sample_count = 262_144
const bin_count = 64
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
	}
	mut samples := []f64{len: repetitions}
	mut checksum := 0.0
	for repetition in 0 .. repetitions {
		started := time.sys_mono_now()
		result := stats.histogram2d_density_edges[f64, f64](x, y, edges, edges)!
		samples[repetition] = f64(time.sys_mono_now() - started) / 1_000_000.0
		for index in 0 .. bin_count * bin_count {
			checksum += result.counts.get_nth(index) * f64(index + 1)
		}
	}
	samples.sort()
	println('operation=histogram2d_density samples=262144 bins=64x64')
	println('median_ms=' + samples[samples.len / 2].str() + ' weighted_checksum=' + (checksum / f64(repetitions)).str())
}
