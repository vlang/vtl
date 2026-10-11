import math
import math.complex as cmplx
import vtl
import vtl.la

fn main() {
	a := vtl.from_2d[cmplx.Complex]([
		[cmplx.complex(1, 1), cmplx.complex(2, 0)],
		[cmplx.complex(2, 0), cmplx.complex(1, -1)],
		[cmplx.complex(1, 0), cmplx.complex(-1, 1)],
	])!
	b := vtl.from_1d[cmplx.Complex]([
		cmplx.complex(-3, 4),
		cmplx.complex(1.5, 5.5),
		cmplx.complex(1.5, 0.5),
	])!

	x, residuals, rank, singular_values := la.lstsq_complex(a, b)!
	assert math.abs(x.get_nth(0).re - 1) < 1e-10
	assert math.abs(x.get_nth(0).im - 2) < 1e-10
	assert math.abs(x.get_nth(1).re + 1) < 1e-10
	assert math.abs(x.get_nth(1).im - 0.5) < 1e-10
	assert rank == 2
	assert residuals.get_nth(0) < 1e-20
	assert singular_values.size == 2

	println('Solution: ${x.to_array()}')
	println('Squared residual sum: ${residuals.get_nth(0)}')
	println('Rank: ${rank}; singular values: ${singular_values.to_array()}')
}
