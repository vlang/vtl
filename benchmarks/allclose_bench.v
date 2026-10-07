// Compare allocation-free allclose with materializing isclose followed by all.
// Run from ~/.vmodules with `v -prod run ./vtl/benchmarks/allclose_bench.v`.
module main

import time
import vtl

fn main() {
	n := 10_000
	iterations := 3
	mut values := []f64{len: n}
	for i in 0 .. n {
		values[i] = f64(i % 997) / 997.0
	}
	a := vtl.from_1d(values)!
	b := vtl.from_1d(values.clone())!
	mut changed_values := values.clone()
	changed_values[0] += 1.0
	changed := vtl.from_1d(changed_values)!
	println('elements,case,method,mean_ms,result')
	bench(a, b, 'equal', iterations)!
	bench(a, changed, 'first_element_differs', iterations)!
}

fn bench(a &vtl.Tensor[f64], b &vtl.Tensor[f64], case_name string, iterations int) ! {
	mut materialized_matches := 0
	mut started := time.sys_mono_now()
	for _ in 0 .. iterations {
		if a.isclose(b)!.all() {
			materialized_matches++
		}
	}
	materialized_ms := f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
	mut direct_matches := 0
	started = time.sys_mono_now()
	for _ in 0 .. iterations {
		if a.allclose(b)! {
			direct_matches++
		}
	}
	direct_ms := f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
	assert materialized_matches == direct_matches
	println('${a.size},${case_name},isclose+all,${materialized_ms:.3f},${materialized_matches}')
	println('${a.size},${case_name},allclose,${direct_ms:.3f},${direct_matches}')
}
