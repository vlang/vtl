module main

import vtl
import vtl.la
import vtl.benchmark

// Benchmark sizes for matrix multiplication
const bench_sizes = [64, 128, 256]

// Repeats per benchmark
const bench_reps = 3

fn main() {
	println('VTL Backend Benchmark')
	println('=====================')
	println('Operations: matmul')
	println('Sizes: ${benchmark.bench_sizes}')
	println('Reps per size: ${benchmark.bench_reps}')
	println('')

	mut results := []benchmark.BenchResult{}

	for sz in benchmark.bench_sizes {
		// Build random-ish f64 matrices
		a_data := build_matrix_data(sz, sz, 1.0)
		b_data := build_matrix_data(sz, sz, 2.0)
		a := vtl.from_2d[f64](a_data) or { panic(err) }
		b := vtl.from_2d[f64](b_data) or { panic(err) }

		// --- CPU ---
		cpu_ns := bench_cpu_matmul(a, b)

		// --- Vulkan ---
		vulkan_ns := bench_vulkan_matmul(a, b)

		// --- VCL ---
		vcl_ns := bench_vcl_matmul(a, b)

		r := benchmark.BenchResult{
			name:      'matmul'
			size:      '${sz}x${sz}'
			cpu_ns:    cpu_ns
			vulkan_ns: vulkan_ns
			vcl_ns:    vcl_ns
		}
		results << r
		println(r.str())
	}
}

fn bench_cpu_matmul(a &vtl.Tensor[f64], b &vtl.Tensor[f64]) i64 {
	mut total := i64(0)
	for _ in 0 .. benchmark.bench_reps {
		total += benchmark.timeit(fn [a, b] () ! {
			la.matmul[f64](a, b)!
		})
	}
	return total / benchmark.bench_reps
}

fn bench_vulkan_matmul(a &vtl.Tensor[f64], b &vtl.Tensor[f64]) i64 {
	return bench_vulkan_matmul_impl(a, b)
}

fn bench_vcl_matmul(a &vtl.Tensor[f64], b &vtl.Tensor[f64]) i64 {
	return bench_vcl_matmul_impl(a, b)
}

// build_matrix_data generates a sz×sz 2D slice filled with seed-scaled values
fn build_matrix_data(rows int, cols int, seed f64) [][]f64 {
	mut data := [][]f64{len: rows}
	for r in 0 .. rows {
		mut row := []f64{len: cols}
		for c in 0 .. cols {
			row[c] = seed * f64(r * cols + c + 1) / f64(rows * cols)
		}
		data[r] = row
	}
	return data
}
