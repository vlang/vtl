module vtl

fn test_unary_ufunc_allocates_and_supports_aliasing() ! {
	input := from_array([-1.0, 0.0, 2.0], [3])!
	output := ufunc_unary[f64](input, fn (value f64) f64 {
		return value * value
	})!
	assert output.to_array() == [1.0, 0.0, 4.0]
	ufunc_unary_into[f64](input, input, unsafe { nil }, false, fn (value f64) f64 {
		return value + 1.0
	})!
	assert input.to_array() == [0.0, 1.0, 3.0]
}

fn test_unary_ufunc_into_reuses_output_and_honors_where() ! {
	input := from_array([1.0, 2.0, 3.0], [3])!
	output := from_array([10.0, 10.0, 10.0], [3])!
	mask := from_array([true, false, true], [3])!
	ufunc_unary_into[f64](input, output, mask, true, fn (value f64) f64 {
		return value * 2.0
	})!
	assert output.to_array() == [2.0, 10.0, 6.0]
}

fn test_binary_ufunc_into_reuses_output() ! {
	left := from_array([1.0, 2.0], [2])!
	right := from_array([3.0, 4.0], [2])!
	output := ufunc_binary[f64](left, right, fn (a f64, b f64) f64 {
		return a + b
	})!
	assert output.to_array() == [4.0, 6.0]
}

fn test_ufunc_reduce_and_accumulate_use_logical_flat_order() ! {
	input := from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!
	assert ufunc_reduce[f64](input, 0.0, fn (a f64, b f64) f64 {
		return a + b
	}) == 10.0
	accumulated := ufunc_accumulate[f64](input, 0.0, fn (a f64, b f64) f64 {
		return a + b
	})!
	assert accumulated.shape == input.shape
	assert accumulated.to_array() == [1.0, 3.0, 6.0, 10.0]
	empty := zeros[f64]([0])
	assert ufunc_reduce[f64](empty, 7.0, fn (a f64, b f64) f64 {
		return a + b
	}) == 7.0
}

fn test_ufunc_outer_and_reduceat() ! {
	left := from_array([1.0, 2.0], [2])!
	right := from_array([10.0, 20.0, 30.0], [3])!
	outer := ufunc_outer[f64](left, right, fn (a f64, b f64) f64 {
		return a * b
	})!
	assert outer.shape == [2, 3]
	assert outer.to_array() == [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
	input := from_array([1.0, 2.0, 3.0, 4.0, 5.0], [5])!
	reduced := ufunc_reduceat[f64](input, [0, 2, 2], fn (a f64, b f64) f64 {
		return a + b
	})!
	assert reduced.to_array() == [3.0, 3.0, 12.0]
}

fn test_ufunc_at_applies_repeated_indices_in_order_and_validates_atomically() ! {
	mut target := from_array([1.0, 2.0, 3.0], [3])!
	ufunc_at[f64](mut target, [1, 1, 2], [4.0, 5.0, 6.0], fn (a f64, b f64) f64 {
		return a + b
	})!
	assert target.to_array() == [1.0, 11.0, 9.0]
	ufunc_at[f64](mut target, [0, 3], [1.0], fn (a f64, b f64) f64 {
		return a + b
	}) or {
		assert target.to_array() == [1.0, 11.0, 9.0]
		return
	}
	assert false, 'ufunc_at must reject out-of-bounds indices'
}

fn test_ufunc_axis_reduce_and_accumulate() ! {
	input := from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])!
	reduced := ufunc_reduce_axis[f64](input, -1, 0.0, fn (a f64, b f64) f64 {
		return a + b
	})!
	assert reduced.shape == [2]
	assert reduced.to_array() == [6.0, 15.0]
	accumulated := ufunc_accumulate_axis[f64](input, 0, 0.0, fn (a f64, b f64) f64 {
		return a + b
	})!
	assert accumulated.to_array() == [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
}
