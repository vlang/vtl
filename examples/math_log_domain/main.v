module main

import math
import vtl

fn main() {
	log_weights := vtl.from_1d[f64]([-1000.0, -1001.0, -999.5]) or { panic(err) }
	reference := vtl.from_1d[f64]([-1000.0]) or { panic(err) }
	combined := vtl.logaddexp(log_weights, reference) or { panic(err) }
	probability_signs := vtl.from_1d([0.0, -0.0, -3.0, math.inf(-1)]) or { panic(err) }
	signed_weights := vtl.copysign(log_weights.abs(), log_weights) or { panic(err) }

	println('log weights: ${log_weights.to_array()}')
	println('stable log sums: ${combined.to_array()}')
	println('copysign: ${signed_weights.to_array()}')
	println('sign bits: ${probability_signs.signbit() or { panic(err) }.to_array()}')
	println('base-2 log sum: ${vtl.logaddexp2(log_weights, reference) or { panic(err) }.to_array()}')
}
