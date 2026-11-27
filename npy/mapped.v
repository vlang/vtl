module npy

import os
import vtl

// MappedNpy provides read-only, row-wise access to a mapped .npy file.
// Each returned tensor owns its decoded values and remains valid after close.
@[heap]
pub struct MappedNpy[T] {
mut:
	mapping       &os.Mmap = unsafe { nil }
	path          string
	mapped_length u64
	info          NpyFileInfo
	closed        bool
}

// open_mapped maps a supported .npy file without copying or decoding its payload.
// Call close when finished; the mapped file must not be modified or truncated
// while it is open.
pub fn open_mapped[T](path string) !&MappedNpy[T] {
	expected := npy_type[T]() or { return err }
	mut mapping := os.mmap_open(path, .read_only)!
	mut keep_mapping := false
	defer {
		if !keep_mapping {
			mapping.close()
		}
	}
	bytes := unsafe { mapping.bytes() }
	info := parse_npy_info(bytes, expected)!
	keep_mapping = true
	return &MappedNpy[T]{
		mapping:       mapping
		path:          path
		mapped_length: mapping.len()
		info:          info
	}
}

// shape returns a copy of the mapped array's shape.
pub fn (mapped &MappedNpy[T]) shape() []int {
	return mapped.info.shape.clone()
}

// size returns the total number of logical elements.
pub fn (mapped &MappedNpy[T]) size() int {
	return mapped.info.count
}

// is_closed reports whether close has released the mapped region.
pub fn (mapped &MappedNpy[T]) is_closed() bool {
	return mapped.closed
}

// read_rows decodes rows [start, end) along the leading axis into owned storage.
pub fn (mapped &MappedNpy[T]) read_rows(start int, end int) !&vtl.Tensor[T] {
	if mapped.closed {
		return error('npy.read_rows: mapping is closed')
	}
	if mapped.info.shape.len == 0 {
		return error('npy.read_rows: scalar arrays do not have rows')
	}
	if start < 0 || end < start || end > mapped.info.shape[0] {
		return error('npy.read_rows: row range [${start}, ${end}) is out of bounds')
	}
	if os.file_size(mapped.path) != mapped.mapped_length {
		return error('npy.read_rows: mapped file size changed while open')
	}
	mut output_shape := mapped.info.shape.clone()
	output_shape[0] = end - start
	mut result := vtl.empty[T](output_shape, memory: .row_major)
	if result.size == 0 {
		return result
	}
	row_size := element_count(mapped.info.shape[1..])!
	first_element := start * row_size
	bytes := unsafe { mapped.mapping.bytes() }
	for local_index in 0 .. result.size {
		logical_index := first_element + local_index
		source_index := if mapped.info.fortran_order {
			fortran_index(logical_index, mapped.info.shape)
		} else {
			logical_index
		}
		offset := mapped.info.data_offset + source_index * mapped.info.descriptor.width
		value := npy_value_from_bytes[T](bytes, offset, mapped.info.descriptor.width,
			mapped.info.little_endian)
		result.set_nth[T](local_index, value)
	}
	return result
}

// close releases the mapping. Decoded tensors returned by read_rows remain valid.
// Closing more than once is safe.
pub fn (mut mapped MappedNpy[T]) close() {
	if mapped.closed {
		return
	}
	mapped.mapping.close()
	mapped.closed = true
}
