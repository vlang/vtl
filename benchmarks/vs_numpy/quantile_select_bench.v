// Compile with -prod from ~/.vmodules, then execute the binary separately.
module main

import time
import vtl
import vtl.stats
import vtl.benchmarks.util as bu

fn main() {
	bu.print_header('VTL scalar median selection benchmark (f64)')
	println('Implementation | Size | Avg (ms) | checksum')
	println('---------------+------+----------+---------')
	for n in [100_000, 1_000_000] {
		mut data := []f64{len: n}
		for i in 0 .. n {
			data[i] = f64((i * 7919 + 17) % 1009) / 1009.0
		}
		tensor := vtl.from_array(data, [n])!
		bench_selection('vtl_select', tensor) or { panic(err) }
		bench_full_sort('vtl_full_sort', data)
	}
	println('\nNumPy reference: numpy_quantile_baseline.py')
}

fn bench_selection(name string, tensor &vtl.Tensor[f64]) ! {
	for _ in 0 .. 2 {
		_ = stats.quantile_with_method(tensor, 0.5, .linear)!
	}
	mut samples := []f64{len: 7}
	mut checksum := 0.0
	for i in 0 .. samples.len {
		started := time.sys_mono_now()
		checksum += stats.quantile_with_method(tensor, 0.5, .linear)!
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
	}
	println('${name} | ${tensor.size} | ${bu.mean_time_ms(mut samples):.4f} | ${checksum:.7f}')
}

fn bench_full_sort(name string, data []f64) {
	mut samples := []f64{len: 7}
	mut checksum := 0.0
	for i in 0 .. samples.len {
		started := time.sys_mono_now()
		mut sorted := data.clone()
		sorted.sort()
		checksum += (sorted[data.len / 2 - 1] + sorted[data.len / 2]) / 2.0
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
	}
	println('${name} | ${data.len} | ${bu.mean_time_ms(mut samples):.4f} | ${checksum:.7f}')
}
