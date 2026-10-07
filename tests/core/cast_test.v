module core

import vtl

fn test_tensor_data_type_casts_between_numeric_variants() {
	assert vtl.td(u8(200)).int() == 200
	assert vtl.td(u16(300)).i64() == 300
	assert vtl.td(u32(40000)).f64() == 40000.0
	assert vtl.td(u64(50000)).f32() == 50000.0
	assert vtl.td(i8(-12)).i16() == -12
	assert vtl.td(i16(1234)).u32() == 1234
	assert vtl.td(i32(-123456)).i64() == -123456
	assert vtl.td(i16(321)).i32() == 321
	assert vtl.td(f64(9.75)).int() == 9
	assert vtl.td(f32(7.5)).u16() == 7
	assert vtl.td(true).int() == 1
	assert vtl.td(false).f64() == 0.0
}

fn test_tensor_as_numeric_types_preserves_shape_and_logical_order() ! {
	tensor := vtl.from_array[u32]([1, 2, 3, 4], [2, 2])!
	assert tensor.as_i8().to_array() == [1, 2, 3, 4]
	assert tensor.as_i16().to_array() == [1, 2, 3, 4]
	assert tensor.as_i32().to_array() == [1, 2, 3, 4]
	assert tensor.as_i64().to_array() == [1, 2, 3, 4]
	assert tensor.as_int().to_array() == [1, 2, 3, 4]
	assert tensor.as_u8().to_array() == [1, 2, 3, 4]
	assert tensor.as_u16().to_array() == [1, 2, 3, 4]
	assert tensor.as_u32().to_array() == [1, 2, 3, 4]
	assert tensor.as_u64().to_array() == [1, 2, 3, 4]
	assert tensor.as_f32().to_array() == [f32(1.0), 2.0, 3.0, 4.0]
	assert tensor.as_f64().to_array() == [1.0, 2.0, 3.0, 4.0]

	view := tensor.t()!
	cast_view := view.as_f64()
	assert cast_view.shape == [2, 2]
	assert cast_view.to_array() == [1.0, 3.0, 2.0, 4.0]
	assert cast_view.is_row_major_contiguous()
}

fn test_tensor_as_casts_float_values_and_boolean_values() ! {
	tf := vtl.from_1d([1.75, 0.0])!
	assert tf.as_i64().to_array() == [1, 0]
	assert tf.as_u16().to_array() == [1, 0]
	assert tf.as_bool().to_array() == [true, false]

	booleans := vtl.from_1d([true, false])!
	assert booleans.as_int().to_array() == [1, 0]
	assert booleans.as_f64().to_array() == [1.0, 0.0]
}
