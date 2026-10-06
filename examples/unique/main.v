module main

import vtl

fn main() {
	samples := vtl.from_1d([4, 2, 4, 1, 2, 4])!
	values := vtl.unique(samples)!
	counts := vtl.unique_counts(samples)!

	println('Unique values: ${values.to_array()}')
	println('Sorted values: ${counts.values.to_array()}')
	println('Occurrences: ${counts.counts.to_array()}')
}
