module core

import vtl

fn test_dtype_of_and_tensor_dtype_report_element_types() {
	assert vtl.dtype_of[bool]() == .boolean
	assert vtl.dtype_of[i8]() == .int8
	assert vtl.dtype_of[i16]() == .int16
	assert vtl.dtype_of[i32]() == .int32
	assert vtl.dtype_of[i64]() == .int64
	assert vtl.dtype_of[int]() == .native_int
	assert vtl.dtype_of[u8]() == .uint8
	assert vtl.dtype_of[u16]() == .uint16
	assert vtl.dtype_of[u32]() == .uint32
	assert vtl.dtype_of[u64]() == .uint64
	assert vtl.dtype_of[f32]() == .float32
	assert vtl.dtype_of[f64]() == .float64
	assert vtl.dtype_of[string]() == .string

	tensor := vtl.from_1d([1, 2, 3])!
	assert tensor.dtype() == .native_int
	tensor32 := vtl.from_1d[i32]([1, 2, 3])!
	assert tensor32.dtype() == .int32
	assert tensor32.add(tensor32)!.to_array() == [i32(2), 4, 6]
}

fn test_promote_types_for_matching_numeric_kinds() {
	assert vtl.promote_types(.int8, .int16)! == .int16
	assert vtl.promote_types(.int16, .uint16)! == .int32
	assert vtl.promote_types(.uint8, .uint16)! == .uint16
	assert vtl.promote_types(.int16, .uint8)! == .int16
	assert vtl.promote_types(.uint8, .int8)! == .int16
	assert vtl.promote_types(.int32, .uint16)! == .int32
	assert vtl.promote_types(.uint32, .int64)! == .int64
	assert vtl.promote_types(.uint64, .int64)! == .float64
	assert vtl.promote_types(.float32, .uint16)! == .float32
	assert vtl.promote_types(.float32, .uint32)! == .float64
	assert vtl.promote_types(.float32, .native_int)! == .float64
	assert vtl.promote_types(.boolean, .uint8)! == .uint8
	assert vtl.promote_types(.float32, .float64)! == .float64
	assert vtl.promote_types(.boolean, .boolean)! == .boolean
	assert vtl.promote_types(.string, .string)! == .string
}

fn test_promote_types_rejects_string_and_numeric_mixtures() {
	if _ := vtl.promote_types(.string, .int8) {
		assert false, 'promote_types must reject string/numeric mixtures'
	} else {
		assert true
	}
}
