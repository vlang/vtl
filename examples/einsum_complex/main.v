module main

import math.complex as cmplx
import vtl

fn main() {
	a := vtl.from_2d[cmplx.Complex]([
		[cmplx.complex(1.0, 1.0), cmplx.complex(2.0, 0.0)],
		[cmplx.complex(3.0, -1.0), cmplx.complex(4.0, 0.0)],
	]) or { panic(err) }
	b := vtl.from_2d[cmplx.Complex]([
		[cmplx.complex(0.0, 1.0), cmplx.complex(2.0, 0.0)],
		[cmplx.complex(1.0, 0.0), cmplx.complex(0.0, -1.0)],
	]) or { panic(err) }
	product := vtl.einsum[cmplx.Complex]('ij,jk->ik', a, b) or { panic(err) }

	println('Complex contraction result: ${product}')
	assert product.to_array() == [
		cmplx.complex(1.0, 1.0),
		cmplx.complex(2.0, 0.0),
		cmplx.complex(5.0, 3.0),
		cmplx.complex(6.0, -6.0),
	]
}
