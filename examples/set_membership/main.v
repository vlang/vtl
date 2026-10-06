module main

import vtl

fn main() {
	measurements := vtl.from_1d([12, 7, 18, 4, 12, 9])!
	accepted := vtl.from_1d([4, 12, 18])!
	mask := vtl.isin(measurements, accepted)
	selected := measurements.masked_select(mask)!

	println('Membership mask: ${mask.to_array()}')
	println('Accepted measurements: ${selected.to_array()}')
}
