module main

import vtl

fn main() {
	values := vtl.from_1d[f64]([1.25, -2.75, 0.75, -8.0]) or { panic(err) }
	fractional, integral := values.modf() or { panic(err) }
	mantissa, exponent := values.frexp() or { panic(err) }
	reconstructed := vtl.ldexp(mantissa, exponent) or { panic(err) }

	println('values:       ${values.to_array()}')
	println('fractional:   ${fractional.to_array()}')
	println('integral:     ${integral.to_array()}')
	println('mantissa:     ${mantissa.to_array()}')
	println('exponent:     ${exponent.to_array()}')
	println('reconstructed:${reconstructed.to_array()}')
}
