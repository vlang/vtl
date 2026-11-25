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

fn test_binary_ufunc_broadcasts_operands_and_where_mask() ! {
	left := from_array([1.0, 2.0], [2, 1])!
	right := from_array([10.0, 20.0, 30.0], [1, 3])!
	output := from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 3])!
	mask := from_array([true, false, true], [1, 3])!
	ufunc_binary_into[f64](left, right, output, mask, true, fn (a f64, b f64) f64 {
		return a + b
	})!
	assert output.to_array() == [11.0, 0.0, 31.0, 12.0, 0.0, 32.0]
	broadcast_output := ufunc_binary[f64](left, right, fn (a f64, b f64) f64 {
		return a + b
	})!
	assert broadcast_output.shape == [2, 3]
	assert broadcast_output.to_array() == [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
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

fn test_ufunc_views_and_overlapping_output_use_logical_order() ! {
	mut storage := from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!
	mut transposed_output := storage.transpose([1, 0])!
	ufunc_unary_into[f64](storage, transposed_output, unsafe { nil }, false, fn (value f64) f64 {
		return value + 10.0
	})!
	assert transposed_output.to_array() == [11.0, 12.0, 13.0, 14.0]
	assert storage.to_array() == [11.0, 13.0, 12.0, 14.0]

	mut binary_storage := from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!
	mut binary_output := binary_storage.transpose([1, 0])!
	right := from_array([10.0, 20.0, 30.0, 40.0], [2, 2])!
	ufunc_binary_into[f64](binary_storage, right, binary_output, unsafe { nil }, false,
		fn (a f64, b f64) f64 {
			return a + b
		})!
	assert binary_output.to_array() == [11.0, 22.0, 33.0, 44.0]
	assert binary_storage.to_array() == [11.0, 33.0, 22.0, 44.0]

	transposed := from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])!.transpose([1, 0])!
	squared := ufunc_unary[f64](transposed, fn (value f64) f64 {
		return value * value
	})!
	assert squared.shape == [3, 2]
	assert squared.to_array() == [1.0, 16.0, 4.0, 25.0, 9.0, 36.0]
	assert ufunc_reduce[f64](transposed, 0.0, fn (a f64, b f64) f64 {
		return a * 10.0 + b
	}) == 142536
}

fn test_ufunc_empty_inputs_preserve_identity_and_shapes() ! {
	empty := zeros[f64]([0, 2])
	unary := ufunc_unary[f64](empty, fn (value f64) f64 {
		return value + 1.0
	})!
	assert unary.shape == [0, 2]
	assert unary.size() == 0
	binary := ufunc_binary[f64](empty, ones[f64]([1, 2]), fn (a f64, b f64) f64 {
		return a + b
	})!
	assert binary.shape == [0, 2]
	assert ufunc_reduce[f64](empty, 7.0, fn (a f64, b f64) f64 {
		return a + b
	}) == 7.0
	reduced := ufunc_reduce_axis[f64](empty, 0, 7.0, fn (a f64, b f64) f64 {
		return a + b
	})!
	assert reduced.shape == [2]
	assert reduced.to_array() == [7.0, 7.0]
	accumulated := ufunc_accumulate_axis[f64](empty, 0, 0.0, fn (a f64, b f64) f64 {
		return a + b
	})!
	assert accumulated.shape == [0, 2]
	if _ := ufunc_reduceat[f64](empty, [0], fn (a f64, b f64) f64 {
		return a + b
	}) {
		assert false, 'reduceat must reject indices into empty inputs'
	}
}

fn test_ufunc_into_reports_shape_and_mask_errors() {
	left := from_array([1.0, 2.0], [2])!
	right := from_array([3.0, 4.0], [2])!
	wrong_output := zeros[f64]([3])
	if _ := ufunc_binary_into[f64](left, right, wrong_output, unsafe { nil }, false,
		fn (a f64, b f64) f64 {
			return a + b
		}) {
		assert false, 'binary ufunc must reject output shape mismatches'
	}
	output := zeros[f64]([2])
	if _ := ufunc_binary_into[f64](left, right, output, unsafe { nil }, true,
		fn (a f64, b f64) f64 {
			return a + b
		}) {
		assert false, 'binary ufunc must reject a missing where mask'
	}
}
