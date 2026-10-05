module main

import vtl
import vtl.stats

fn main() {
	values := vtl.from_1d([32.0, 10.0, 20.0, 40.0, 30.0])!
	for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
		println('q=${q}: ${stats.quantile_linear(values, q)!}')
	}
}
