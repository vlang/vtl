module npy

import encoding.binary
import math
import os
import strconv
import vtl

const magic = [u8(0x93), u8(`N`), u8(`U`), u8(`M`), u8(`P`), u8(`Y`)]

struct NpyType {
	kind  u8
	width int
}

struct NpyDescriptor {
	kind   u8
	width  int
	endian u8
}

// write writes a CPU tensor as an uncompressed NumPy .npy v1.0 file.
// Numeric primitive types and row-major logical tensor order are supported.
pub fn write[T](path string, tensor &vtl.Tensor[T]) ! {
	npy_dtype := npy_type[T]() or { return err }
	descriptor := '<${rune(npy_dtype.kind)}${npy_dtype.width}'
	shape := format_shape(tensor.shape)
	mut header := "{'descr': '${descriptor}', 'fortran_order': False, 'shape': ${shape}, }"
	padding := (16 - ((10 + header.len + 1) % 16)) % 16
	header += ' '.repeat(padding)
	header += '\n'
	if header.len > 0xffff {
		return error('npy.write: header exceeds the v1.0 size limit')
	}
	if tensor.size > (max_int - 10 - header.len) / npy_dtype.width {
		return error('npy.write: array size overflows addressable memory')
	}
	mut bytes := []u8{cap: 10 + header.len + tensor.size * npy_dtype.width}
	bytes << magic
	bytes << u8(1)
	bytes << u8(0)
	mut header_length := []u8{len: 2}
	binary.little_endian_put_u16(mut header_length, u16(header.len))
	bytes << header_length
	bytes << header.bytes()
	mut iter := tensor.iterator()
	for {
		value, _ := iter.next() or { break }
		append_value[T](mut bytes, value, npy_dtype.width)
	}
	os.write_file_array(path, bytes)!
}

// read reads a NumPy .npy v1.0, v2.0, or v3.0 array into T.
// The file dtype must have the same kind and width as T. Both byte orders and
// Fortran-order files are converted to VTL's row-major logical tensor layout.
pub fn read[T](path string) !&vtl.Tensor[T] {
	bytes := os.read_bytes(path)!
	if bytes.len < 10 || bytes[..6] != magic {
		return error('npy.read: invalid NumPy magic prefix')
	}
	major := bytes[6]
	minor := bytes[7]
	if major < 1 || major > 3 || minor != 0 {
		return error('npy.read: unsupported .npy version ${major}.${minor}')
	}
	mut header_length := 0
	mut data_offset := 0
	if major == 1 {
		raw_header_length := binary.little_endian_u16(bytes[8..])
		if int(raw_header_length) > bytes.len - 10 {
			return error('npy.read: truncated header')
		}
		header_length = int(raw_header_length)
		data_offset = 10 + header_length
	} else {
		if bytes.len < 12 {
			return error('npy.read: truncated version ${major} header')
		}
		raw_header_length := binary.little_endian_u32(bytes[8..])
		if u64(raw_header_length) > u64(bytes.len - 12) {
			return error('npy.read: truncated header')
		}
		header_length = int(raw_header_length)
		data_offset = 12 + header_length
	}
	if data_offset < 0 || data_offset > bytes.len {
		return error('npy.read: truncated header')
	}
	header := bytes[if major == 1 { 10 } else { 12 }..data_offset].bytestr()
	descriptor := parse_descriptor(header)!
	expected := npy_type[T]() or { return err }
	if descriptor.kind != expected.kind || descriptor.width != expected.width {
		return error('npy.read: dtype ${rune(descriptor.kind)}${descriptor.width} does not match requested V type')
	}
	shape := parse_shape(header)!
	count := element_count(shape)!
	if count > (max_int - data_offset) / descriptor.width {
		return error('npy.read: array size overflows addressable memory')
	}
	byte_count := count * descriptor.width
	if bytes.len - data_offset < byte_count {
		return error('npy.read: truncated data payload')
	}
	fortran := parse_fortran_order(header)!
	mut little_endian := descriptor.endian != u8(`>`)
	if descriptor.endian == u8(`=`) {
		little_endian = host_is_little_endian()
	}
	mut values := []T{len: count}
	for i in 0 .. count {
		source_index := if fortran { fortran_index(i, shape) } else { i }
		offset := data_offset + source_index * descriptor.width
		bits := read_bits(bytes, offset, descriptor.width, little_endian)
		values[i] = value_from_bits[T](bits)
	}
	return vtl.from_array[T](values, shape)!
}

fn npy_type[T]() !NpyType {
	$if T is bool {
		return NpyType{u8(`b`), 1}
	} $else $if T is f32 {
		return NpyType{u8(`f`), 4}
	} $else $if T is f64 {
		return NpyType{u8(`f`), 8}
	} $else $if T is i8 {
		return NpyType{u8(`i`), 1}
	} $else $if T is i16 {
		return NpyType{u8(`i`), 2}
	} $else $if T is i32 {
		return NpyType{u8(`i`), 4}
	} $else $if T is i64 {
		return NpyType{u8(`i`), 8}
	} $else $if T is int {
		return NpyType{u8(`i`), sizeof(T)}
	} $else $if T is u8 {
		return NpyType{u8(`u`), 1}
	} $else $if T is u16 {
		return NpyType{u8(`u`), 2}
	} $else $if T is u32 {
		return NpyType{u8(`u`), 4}
	} $else $if T is u64 {
		return NpyType{u8(`u`), 8}
	} $else {
		return error('npy: unsupported tensor element type ${T.name}')
	}
}

fn append_value[T](mut output []u8, value T, width int) {
	mut bits := u64(0)
	$if T is f64 {
		bits = math.f64_bits(value)
	} $else $if T is f32 {
		bits = u64(math.f32_bits(value))
	} $else $if T is bool {
		bits = if value { 1 } else { 0 }
	} $else {
		bits = u64(value)
	}
	mut encoded := []u8{len: width}
	match width {
		1 { encoded[0] = u8(bits) }
		2 { binary.little_endian_put_u16(mut encoded, u16(bits)) }
		4 { binary.little_endian_put_u32(mut encoded, u32(bits)) }
		8 { binary.little_endian_put_u64(mut encoded, bits) }
		else {}
	}
	output << encoded
}

fn value_from_bits[T](bits u64) T {
	$if T is f64 {
		return math.f64_from_bits(bits)
	} $else $if T is f32 {
		return math.f32_from_bits(u32(bits))
	} $else $if T is bool {
		return bits != 0
	} $else $if T is i8 {
		return i8(u8(bits))
	} $else $if T is i16 {
		return i16(u16(bits))
	} $else $if T is i32 {
		return i32(u32(bits))
	} $else $if T is i64 {
		return i64(bits)
	} $else $if T is int {
		if sizeof(T) == 4 {
			return int(i32(u32(bits)))
		} else {
			return int(i64(bits))
		}
	} $else $if T is u8 {
		return u8(bits)
	} $else $if T is u16 {
		return u16(bits)
	} $else $if T is u32 {
		return u32(bits)
	} $else $if T is u64 {
		return bits
	} $else {
		panic('npy: unsupported tensor element type ${T.name}')
	}
}

fn read_bits(bytes []u8, offset int, width int, little_endian bool) u64 {
	mut result := u64(0)
	if little_endian {
		for i in 0 .. width {
			result |= u64(bytes[offset + i]) << (8 * i)
		}
	} else {
		for i in 0 .. width {
			result = (result << 8) | u64(bytes[offset + i])
		}
	}
	return result
}

fn format_shape(shape []int) string {
	if shape.len == 0 {
		return '()'
	}
	mut dimensions := shape.map(it.str()).join(', ')
	if shape.len == 1 {
		dimensions += ','
	}
	return '(${dimensions})'
}

fn element_count(shape []int) !int {
	mut has_zero_dimension := false
	for dimension in shape {
		if dimension < 0 {
			return error('npy.read: negative dimensions are invalid')
		}
		if dimension == 0 {
			has_zero_dimension = true
		}
	}
	if has_zero_dimension {
		return 0
	}
	mut count := 1
	for dimension in shape {
		if count > max_int / dimension {
			return error('npy.read: shape size overflows addressable memory')
		}
		count *= dimension
	}
	return count
}

fn fortran_index(index int, shape []int) int {
	mut coordinates := []int{len: shape.len}
	mut remaining := index
	for axis := shape.len - 1; axis >= 0; axis-- {
		coordinates[axis] = remaining % shape[axis]
		remaining /= shape[axis]
	}
	mut f_stride := 1
	mut result := 0
	for axis in 0 .. shape.len {
		dimension := shape[axis]
		result += coordinates[axis] * f_stride
		f_stride *= dimension
	}
	return result
}

fn host_is_little_endian() bool {
	$if little_endian {
		return true
	} $else {
		return false
	}
}

fn parse_descriptor(header string) !NpyDescriptor {
	key := header.index("'descr'") or { return error('npy.read: missing descr field') }
	colon := header[key..].index(':') or { return error('npy.read: malformed descr field') }
	value := header[key + colon + 1..].trim_space()
	if value.len < 4 || value[0] != u8(`'`) {
		return error('npy.read: malformed dtype descriptor')
	}
	end := value[1..].index("'") or { return error('npy.read: unterminated dtype descriptor') }
	descriptor := value[1..end + 1]
	if descriptor.len < 3 {
		return error('npy.read: invalid dtype descriptor')
	}
	endian := descriptor[0]
	kind := descriptor[1]
	if endian !in [`<`, `>`, `=`, `|`] || kind !in [`b`, `f`, `i`, `u`] {
		return error('npy.read: unsupported dtype descriptor ${descriptor}')
	}
	width := strconv.atoi(descriptor[2..]) or { return error('npy.read: invalid dtype width') }
	if width !in [1, 2, 4, 8] || (endian == `|` && width != 1) || (kind == `b` && width != 1)
		|| (kind == `f` && width !in [4, 8]) {
		return error('npy.read: unsupported dtype width ${width}')
	}
	return NpyDescriptor{kind, width, endian}
}

fn parse_shape(header string) ![]int {
	key := header.index("'shape'") or { return error('npy.read: missing shape field') }
	colon := header[key..].index(':') or { return error('npy.read: malformed shape field') }
	value := header[key + colon + 1..].trim_space()
	if value.len == 0 || value[0] != u8(`(`) {
		return error('npy.read: malformed shape tuple')
	}
	end := value.index(')') or { return error('npy.read: unterminated shape tuple') }
	contents := value[1..end].trim_space()
	mut shape := []int{}
	if contents.len > 0 {
		for part in contents.split(',') {
			dimension := part.trim_space()
			if dimension.len > 0 {
				shape << strconv.atoi(dimension) or { return error('npy.read: invalid shape dimension') }
			}
		}
	}
	return shape
}

fn parse_fortran_order(header string) !bool {
	key := header.index("'fortran_order'") or {
		return error('npy.read: missing fortran_order field')
	}
	colon := header[key..].index(':') or {
		return error('npy.read: malformed fortran_order field')
	}
	value := header[key + colon + 1..].trim_space()
	if value.starts_with('False,') || value.starts_with('False}') {
		return false
	}
	if value.starts_with('True,') || value.starts_with('True}') {
		return true
	}
	return error('npy.read: invalid fortran_order value')
}
