module npy

import encoding.binary
import os
import vtl

fn test_npy_f64_round_trip_preserves_shape_and_values() {
	path := os.join_path(os.temp_dir(), 'vtl_npy_f64_round_trip.npy')
	defer {
		os.rm(path) or {}
	}
	original := vtl.from_array[f64]([1.5, -2.25, 3.0, 4.75], [2, 2])!
	write(path, original)!
	bytes := os.read_bytes(path)!
	header_length := int(binary.little_endian_u16(bytes[8..]))
	assert (10 + header_length) % 64 == 0
	loaded := read[f64](path)!
	assert loaded.shape == [2, 2]
	assert loaded.to_array() == [1.5, -2.25, 3.0, 4.75]
}

fn test_npy_integer_round_trip() {
	path := os.join_path(os.temp_dir(), 'vtl_npy_integer_round_trip.npy')
	defer {
		os.rm(path) or {}
	}
	original := vtl.from_array[i16]([-3, 0, 1024], [3])!
	write(path, original)!
	loaded := read[i16](path)!
	assert loaded.shape == [3]
	assert loaded.to_array() == [-3, 0, 1024]
}

fn test_npy_write_serializes_non_contiguous_views_in_logical_order() {
	path := os.join_path(os.temp_dir(), 'vtl_npy_transposed_view.npy')
	defer {
		os.rm(path) or {}
	}
	original := vtl.from_array[f64]([1, 2, 3, 4, 5, 6], [2, 3])!
	transposed := original.transpose([1, 0])!
	write(path, transposed)!
	loaded := read[f64](path)!
	assert loaded.shape == [3, 2]
	assert loaded.to_array() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
}

fn test_npy_round_trips_all_primitive_numeric_types() {
	check_npy_round_trip[f32]('f32', [f32(0.5), -2.25])
	check_npy_round_trip[i8]('i8', [i8(-128), 127])
	check_npy_round_trip[i32]('i32', [i32(-2147483648), 2147483647])
	check_npy_round_trip[i64]('i64', [i64(-9223372036854775807), 42])
	check_npy_round_trip[int]('int', [int(-7), 9])
	check_npy_round_trip[u8]('u8', [u8(0), 255])
	check_npy_round_trip[u16]('u16', [u16(0), 65535])
	check_npy_round_trip[u64]('u64', [u64(0), 18446744073709551615])
	check_npy_round_trip[bool]('bool', [true, false, true])
}

fn test_npy_native_int_uses_platform_width() {
	dtype := npy_type[int]()!
	assert dtype.width == sizeof(int)
}

fn test_npy_writer_uses_byte_order_independent_descriptor_for_bool() {
	path := os.join_path(os.temp_dir(), 'vtl_npy_bool_descriptor.npy')
	defer {
		os.rm(path) or {}
	}
	write(path, vtl.from_1d([true, false])!)!
	bytes := os.read_bytes(path)!
	header_length := int(binary.little_endian_u16(bytes[8..]))
	header := bytes[10..10 + header_length].bytestr()
	assert header.contains("'descr': '|b1'")
}

fn check_npy_round_trip[T](suffix string, values []T) {
	path := os.join_path(os.temp_dir(), 'vtl_npy_${suffix}_round_trip.npy')
	defer {
		os.rm(path) or {}
	}
	original := vtl.from_array[T](values, [values.len])!
	write(path, original)!
	loaded := read[T](path)!
	assert loaded.shape == [values.len]
	assert loaded.to_array() == values
}

fn test_npy_reads_big_endian_fortran_order_v2() {
	path := os.join_path(os.temp_dir(), 'vtl_npy_big_endian_fortran.npy')
	defer {
		os.rm(path) or {}
	}
	mut payload := []u8{len: 24}
	for i, value in [u32(1), 4, 2, 5, 3, 6] {
		binary.little_endian_put_u32_at(mut payload, value, i * 4)
		payload[i * 4], payload[i * 4 + 1], payload[i * 4 + 2], payload[i * 4 + 3] = payload[i * 4 + 3], payload[i * 4 + 2], payload[i * 4 + 1], payload[i * 4]
	}
	write_test_npy(path, 2, "{'descr': '>u4', 'fortran_order': True, 'shape': (2, 3), }", payload)
	loaded := read[u32](path)!
	assert loaded.shape == [2, 3]
	assert loaded.to_array() == [u32(1), 2, 3, 4, 5, 6]
}

fn test_npy_reads_scalar_and_empty_shapes() {
	scalar_path := os.join_path(os.temp_dir(), 'vtl_npy_scalar.npy')
	defer {
		os.rm(scalar_path) or {}
	}
	scalar := vtl.from_array[f64]([9.25], [])!
	write(scalar_path, scalar)!
	loaded_scalar := read[f64](scalar_path)!
	assert loaded_scalar.shape.len == 0
	assert loaded_scalar.get_nth(0) == 9.25

	empty_path := os.join_path(os.temp_dir(), 'vtl_npy_empty.npy')
	defer {
		os.rm(empty_path) or {}
	}
	empty := vtl.from_array[f32]([]f32{}, [2, 0, 3])!
	write(empty_path, empty)!
	loaded_empty := read[f32](empty_path)!
	assert loaded_empty.shape == [2, 0, 3]
	assert loaded_empty.size == 0
}

fn test_npy_reads_version_3_boolean_data() {
	path := os.join_path(os.temp_dir(), 'vtl_npy_v3_bool.npy')
	defer {
		os.rm(path) or {}
	}
	write_test_npy(path, 3, "{'descr': '|b1', 'fortran_order': False, 'shape': (3,), }", [
		u8(1),
		0,
		1,
	])
	loaded := read[bool](path)!
	assert loaded.to_array() == [true, false, true]
}

fn test_npy_rejects_bad_magic_dtype_and_truncated_payload() {
	bad_magic_path := os.join_path(os.temp_dir(), 'vtl_npy_bad_magic.npy')
	defer {
		os.rm(bad_magic_path) or {}
	}
	os.write_file_array(bad_magic_path, [u8(0), 1, 2, 3, 4, 5, 1, 0, 0, 0])!
	assert_npy_read_error[f64](bad_magic_path)

	dtype_path := os.join_path(os.temp_dir(), 'vtl_npy_dtype_mismatch.npy')
	defer {
		os.rm(dtype_path) or {}
	}
	write_test_npy(dtype_path, 1, "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }", [
		u8(0),
		0,
		128,
		63,
	])
	assert_npy_read_error[f64](dtype_path)

	truncated_path := os.join_path(os.temp_dir(), 'vtl_npy_truncated.npy')
	defer {
		os.rm(truncated_path) or {}
	}
	write_test_npy(truncated_path, 1, "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }", [u8(0)])
	assert_npy_read_error[f64](truncated_path)
}

fn assert_npy_read_error[T](path string) {
	_ := read[T](path) or { return }
	assert false, 'expected read to reject ${path}'
}

fn write_test_npy(path string, major u8, raw_header string, payload []u8) {
	mut header := raw_header
	prefix_length := if major == 1 { 10 } else { 12 }
	header += ' '.repeat((64 - ((prefix_length + header.len + 1) % 64)) % 64)
	header += '\n'
	mut bytes := []u8{}
	bytes << magic
	bytes << major
	bytes << u8(0)
	if major == 1 {
		mut length := []u8{len: 2}
		binary.little_endian_put_u16(mut length, u16(header.len))
		bytes << length
	} else {
		mut length := []u8{len: 4}
		binary.little_endian_put_u32(mut length, u32(header.len))
		bytes << length
	}
	bytes << header.bytes()
	bytes << payload
	os.write_file_array(path, bytes) or { panic(err) }
}
