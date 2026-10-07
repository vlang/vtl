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
}
