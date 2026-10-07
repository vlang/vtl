module vtl

// bitwise_and computes the elementwise bitwise AND with tensor broadcasting.
pub fn (a &Tensor[T]) bitwise_and[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn (values []T, _ []int) T {
		return values[0] & values[1]
	})
}

// bitwise_or computes the elementwise bitwise OR with tensor broadcasting.
pub fn (a &Tensor[T]) bitwise_or[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn (values []T, _ []int) T {
		return values[0] | values[1]
	})
}

// bitwise_xor computes the elementwise bitwise XOR with tensor broadcasting.
pub fn (a &Tensor[T]) bitwise_xor[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn (values []T, _ []int) T {
		return values[0] ^ values[1]
	})
}

// bitwise_invert computes the elementwise one's complement.
pub fn (t &Tensor[T]) bitwise_invert[T]() &Tensor[T] {
	return t.map_values(fn [T](value T) T {
		return ~value
	})
}

// left_shift shifts every integer value left by a scalar number of bits.
pub fn (t &Tensor[T]) left_shift[T](shift int) !&Tensor[T] {
	if shift < 0 || shift >= int(sizeof(T)) * 8 {
		return error('left_shift: shift must be within the integer bit width')
	}
	return t.map_values(fn [shift] [T](value T) T {
		return value << shift
	})
}

// right_shift shifts every integer value right by a scalar number of bits.
pub fn (t &Tensor[T]) right_shift[T](shift int) !&Tensor[T] {
	if shift < 0 || shift >= int(sizeof(T)) * 8 {
		return error('right_shift: shift must be within the integer bit width')
	}
	return t.map_values(fn [shift] [T](value T) T {
		return value >> shift
	})
}
