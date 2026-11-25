module main

import vtl

fn main() {
	x := [-1.0, 0.0, 1.0, 2.0]
	y := [2.0, 1.0, 2.0, 5.0]
	fitted := vtl.polynomial_fit(x, y, 2)!
	println('ascending coefficients: ${fitted.coefficients}')
	println('p(3): ${fitted.eval(3.0)}')
	domain := [f64(-1.0), 2.0]!
	window := [f64(-1.0), 1.0]!
	chebyshev_fit := vtl.chebyshev_fit(x, y, 2, domain, window)!
	println('ascending Chebyshev coefficients: ${chebyshev_fit.coefficients}')
	println('Chebyshev p(3): ${chebyshev_fit.eval(3.0)}')
}
