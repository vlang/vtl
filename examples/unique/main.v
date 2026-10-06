module main

import vtl

fn main() {
	samples := vtl.from_1d([4, 2, 4, 1, 2, 4])!
	values := vtl.unique(samples)!
	counts := vtl.unique_counts(samples)!
	inverse := vtl.unique_inverse(samples)!
	first_indices := vtl.unique_first_indices(samples)!

	println('Unique values: ${values.to_array()}')
	println('Sorted values: ${counts.values.to_array()}')
	println('Occurrences: ${counts.counts.to_array()}')
	println('Inverse indices: ${inverse.to_array()}')
	println('First input indices: ${first_indices.to_array()}')
}
