import math
import vtl

fn main() {
	values := vtl.from_1d([math.nan(), 1.5, math.inf(1), -math.inf(1)])!
	mask := values.is_nan()
	assert mask.to_array() == [true, false, false, false]
	println('NaN mask: ${mask.to_array()}')
}
