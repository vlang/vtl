module main

import math.complex as cmplx
import vtl

fn main() {
	values := vtl.from_1d([
		cmplx.complex(1.0, 2.0),
		cmplx.complex(3.0, -1.0),
	]) or { panic(err) }

	println('dtype: ${values.dtype()}')
	println('sum: ${values.add(values)!.str()}')
	println('product: ${values.multiply(values)!.str()}')
	println('quotient: ${values.divide(values)!.str()}')
	real_values := vtl.real(values) or { panic(err) }
	imaginary_values := vtl.imag(values) or { panic(err) }
	conjugated := vtl.conj(values) or { panic(err) }
	magnitudes := vtl.absolute(values) or { panic(err) }
	println('real: ${real_values.str()}')
	println('imag: ${imaginary_values.str()}')
	println('conjugate: ${conjugated.str()}')
	println('magnitude: ${magnitudes.str()}')
}
