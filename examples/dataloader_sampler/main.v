module main

import vsl.la
import vtl.nn.data

fn main() {
	x := la.Matrix.deep2[f64]([
		[0.0, 0.5],
		[1.0, 1.5],
		[2.0, 2.5],
		[3.0, 3.5],
	])
	labels := [10.0, 11.0, 12.0, 13.0]!
	sampler := data.SubsetRandomSampler.new([1, 3])
	mut loader := data.DataLoader.new_with_sampler[f64](x, labels, 1, sampler)!
	for {
		batch, batch_labels := loader.next() or { break }
		println('sample=${int(batch_labels[0])}, features=${batch.get(0, 0)}, ${batch.get(0, 1)}')
	}
}
