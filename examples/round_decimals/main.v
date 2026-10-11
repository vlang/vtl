module main

import vtl

fn main() {
	values := vtl.from_1d([12.5, 25.0, 125.0, 135.0])!
	println('Rounded to tens (ties to even):')
	println(values.round_decimals(-1).to_array())
}
