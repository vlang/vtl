import math
import math.complex as cmplx
import vtl
import vtl.la

fn main() {
	// The zero leading diagonal forces a row swap during partial pivoting.
	a := vtl.from_array[cmplx.Complex]([
		cmplx.Complex{},
		cmplx.Complex{ re: 1, im: 1 },
		cmplx.Complex{ re: 2 },
		cmplx.Complex{ re: 3, im: -1 },
	], [2, 2]) or { panic(err) }
	b := vtl.from_1d[cmplx.Complex]([
		cmplx.Complex{ re: 1, im: 3 },
		cmplx.Complex{ re: 9, im: -1 },
	]) or { panic(err) }

	x := la.solve_complex(a, b) or { panic(err) }

	assert math.abs(x.get_nth(0).re - 1) < 1e-12
	assert math.abs(x.get_nth(0).im + 1) < 1e-12
	assert math.abs(x.get_nth(1).re - 2) < 1e-12
	assert math.abs(x.get_nth(1).im - 1) < 1e-12
	println('Solution: ${x.to_array()}')
}
