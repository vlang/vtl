module core

import vtl

fn test_bitwise_binary_ops_broadcast_and_preserve_integer_type() ! {
	a := vtl.from_array([12, 10, 7, 3], [2, 2])!
	b := vtl.from_1d([10, 12])!
	assert a.bitwise_and(b)!.to_array() == [8, 8, 2, 0]
	assert a.bitwise_or(b)!.to_array() == [14, 14, 15, 15]
	assert a.bitwise_xor(b)!.to_array() == [6, 6, 13, 15]
	unsigned := vtl.from_1d[u8]([0b1100, 0b1010])!
	assert unsigned.bitwise_and(vtl.from_1d[u8]([0b1010])!)!.to_array() == [0b1000, 0b1010]
}

fn test_bitwise_unary_ops_and_shifts() ! {
	values := vtl.from_1d([0b0011, 0b1100, -8])!
	assert values.bitwise_invert().to_array() == [-4, -13, 7]
	assert values.left_shift(1)!.to_array() == [6, 24, -16]
	assert values.right_shift(1)!.to_array() == [1, 6, -4]
	if _ := values.left_shift(-1) {
		assert false, 'left_shift must reject negative counts'
	}
	if _ := values.right_shift(int(sizeof(int)) * 8) {
		assert false, 'right_shift must reject counts outside the integer width'
	}
}
