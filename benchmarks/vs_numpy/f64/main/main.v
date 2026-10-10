// Low-memory VTL f64 GEMM benchmark. Compile with -prod from ~/.vmodules, then run the binary.
module main

import math
import time
import vtl
import vtl.la

const sizes = [128, 256, 512]
const iterations = 10
const warmup_runs = 3

fn main() {
	println('VTL f64 matmul benchmark (end to end)')
	println('Benchmark | Size | Avg (ms) | Checksum')
	for n in sizes {
		bench_matmul_f64(n)
	}
}

fn bench_matmul_f64(n int) {
	mut a_values := []f64{len: n * n}
	mut b_values := []f64{len: n * n}
	for i in 0 .. n {
		for j in 0 .. n {
			a_values[i * n + j] = f64((i * 17 + j * 13) % 997 + 1) / 997.0
			b_values[i * n + j] = f64((i * 7 + j * 19) % 991 + 1) / 991.0
		}
	}
	a := vtl.from_array(a_values, [n, n]) or { panic(err) }
	b := vtl.from_array(b_values, [n, n]) or { panic(err) }
	for _ in 0 .. warmup_runs {
		_ := la.matmul[f64](a, b) or { panic(err) }
	}
	mut samples := []f64{len: iterations}
	mut checksum := 0.0
	for i in 0 .. iterations {
		started := time.sys_mono_now()
		result := la.matmul[f64](a, b) or { panic(err) }
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
		checksum += result.get_nth[f64](n * n / 2 + n / 2)
	}
	mut expected := 0.0
	row := n / 2
	column := n / 2
	for inner in 0 .. n {
		a_value := f64((row * 17 + inner * 13) % 997 + 1) / 997.0
		b_value := f64((inner * 7 + column * 19) % 991 + 1) / 991.0
		expected += a_value * b_value
	}
	if math.abs(checksum / iterations - expected) > 1e-9 {
		panic('VTL f64 matmul checksum mismatch for ${n}x${n}')
	}
	mut total := 0.0
	for sample in samples {
		total += sample
	}
	println('gemm_f64 | ${n}x${n} | ${total / iterations:.3f} | ${checksum:.6f}')
}
