// Compare allocation-free allclose with materializing isclose followed by all.
// Compile with `-prod` from ~/.vmodules, then run the resulting executable.
module main

import time
import vtl

fn main() {
	n := 10_000
	iterations := 1000
	warmup_iterations := 10
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
	bench(a, b, 'equal', iterations, warmup_iterations)!
	bench(a, changed, 'first_element_differs', iterations, warmup_iterations)!
}

fn bench(a &vtl.Tensor[f64], b &vtl.Tensor[f64], case_name string, iterations int, warmup_iterations int) ! {
	for _ in 0 .. warmup_iterations {
		_ = a.isclose(b)!.all()
		_ = a.allclose(b)!
	}
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
	println('${a.size},${case_name},isclose+all,${materialized_ms:.6f},${materialized_matches}')
	println('${a.size},${case_name},allclose,${direct_ms:.6f},${direct_matches}')
}
