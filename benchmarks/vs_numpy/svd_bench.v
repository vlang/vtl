// End-to-end reduced SVD benchmark. Run from ~/.vmodules with `v -prod run`.
module main

import time
import vtl
import vtl.la as vtl_la

fn main() {
	println('VTL reduced SVD benchmark (f64)')
	println('Size       | Avg (ms) | Singular-value checksum')
	println('-----------+----------+-------------------------')
	for size in [16, 32, 64, 128] {
		bench_svd(size)!
	}
	println('Compare with numpy_svd_baseline.py using identical inputs and sizes.')
}

fn bench_svd(size int) ! {
	mut values := []f64{len: size * size}
	for row in 0 .. size {
		for column in 0 .. size {
			values[row * size + column] = f64((row * 17 + column * 13) % 997 + 1) / 997.0
		}
	}
	input := vtl.from_array(values, [size, size])!
	for _ in 0 .. 2 {
		_, _, _ := vtl_la.svd[f64](input, full_matrices: false)!
	}
	mut elapsed := 0.0
	mut checksum := 0.0
	for _ in 0 .. 5 {
		started := time.sys_mono_now()
		_, singular_values, _ := vtl_la.svd[f64](input, full_matrices: false)!
		elapsed += f64(time.sys_mono_now() - started) / 1_000_000.0
		for value in singular_values.to_array() {
			checksum += value
		}
	}
	println('${size}x${size}    | ${elapsed / 5.0} | ${checksum}')
}
