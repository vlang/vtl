module vtl

// set copies a scalar value into a Tensor at the provided index
[inline]
pub fn (mut t Tensor) set(index []int, val voidptr) {
	mut offset := t.offset(index)
	unsafe {t.data.set(offset, val)}
}
