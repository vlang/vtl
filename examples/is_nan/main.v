import math
import vtl

fn main() {
	values := vtl.from_1d([math.nan(), 1.5, math.inf(1), -math.inf(1)])!
	nan_mask := values.is_nan()
	inf_mask := values.is_inf(0)
	finite_mask := values.is_finite()
	assert nan_mask.to_array() == [true, false, false, false]
	assert inf_mask.to_array() == [false, false, true, true]
	assert finite_mask.to_array() == [false, true, false, false]
	println('NaN mask: ${nan_mask.to_array()}')
	println('Infinity mask: ${inf_mask.to_array()}')
	println('Finite mask: ${finite_mask.to_array()}')
}
