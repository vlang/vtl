// benchmark is a VTL module for measuring and comparing CPU, Vulkan GPU, and
// OpenCL GPU performance on common tensor operations.
//
// Usage:
//   v run benchmark/main.v             # CPU only
//   v -d vulkan run benchmark/main.v   # CPU + Vulkan
//   v -d vcl run benchmark/main.v      # CPU + OpenCL
//   v -d vulkan -d vcl run benchmark/main.v  # all backends
module benchmark

import time

// BenchResult holds timing results for a single operation across backends.
pub struct BenchResult {
pub:
	name      string
	size      string
	cpu_ns    i64
	vulkan_ns i64 // -1 if not available
	vcl_ns    i64 // -1 if not available
}

// str returns a human-readable summary line for a BenchResult.
pub fn (r &BenchResult) str() string {
	mut s := '${r.name} [${r.size}]: CPU=${ns_to_ms(r.cpu_ns)}ms'
	if r.vulkan_ns >= 0 {
		speedup := f64(r.cpu_ns) / f64(r.vulkan_ns)
		s += ' | Vulkan=${ns_to_ms(r.vulkan_ns)}ms (${speedup:.2f}x)'
	}
	if r.vcl_ns >= 0 {
		speedup := f64(r.cpu_ns) / f64(r.vcl_ns)
		s += ' | VCL=${ns_to_ms(r.vcl_ns)}ms (${speedup:.2f}x)'
	}
	return s
}

fn ns_to_ms(ns i64) f64 {
	return f64(ns) / 1_000_000.0
}

// timeit runs fn and returns elapsed nanoseconds.
pub fn timeit(f fn () !) i64 {
	t := time.now()
	f() or {}
	return time.now().unix_nano() - t.unix_nano()
}
