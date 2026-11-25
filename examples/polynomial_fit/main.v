module main

import vtl

fn main() {
	x := [-1.0, 0.0, 1.0, 2.0]
	y := [2.0, 1.0, 2.0, 5.0]
	fitted := vtl.polynomial_fit(x, y, 2)!
	println('ascending coefficients: ${fitted.coefficients}')
	println('p(3): ${fitted.eval(3.0)}')
}
