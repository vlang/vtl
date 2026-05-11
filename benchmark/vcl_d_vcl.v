module main

import vtl
import vtl.la
import vtl.benchmark

fn bench_vcl_matmul_impl(a &vtl.Tensor[f64], b &vtl.Tensor[f64]) i64 {
	mut total := i64(0)
	for _ in 0 .. benchmark.bench_reps {
		total += benchmark.timeit(fn [a, b] () ! {
			la.matmul_vcl[f64](a, b)!
		})
	}
	return total / benchmark.bench_reps
}
