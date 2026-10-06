module npz

import compress.szip
import vtl
import vtl.npy

// NpyArray encodes one tensor as a NumPy .npy payload when the archive is
// written, so a mixed-dtype write only materializes one member at a time.
pub interface NpyArray {
	to_npy() ![]u8
}

struct TensorArray[T] {
	tensor &vtl.Tensor[T]
}

fn (array TensorArray[T]) to_npy() ![]u8 {
	return npy.to_bytes[T](array.tensor)!
}

// array wraps one tensor for a mixed-dtype write_arrays call.
pub fn array[T](tensor &vtl.Tensor[T]) NpyArray {
	return TensorArray[T]{
		tensor: tensor
	}
}

// write stores one or more same-element-type tensors in a NumPy .npz archive.
// The archive member name is the map key with a .npy suffix appended when it
// is not already present.
pub fn write[T](path string, arrays map[string]&vtl.Tensor[T]) ! {
	names := sorted_member_names(arrays.keys())!
	mut archive := szip.open(path, .default_compression, .write)!
	defer {
		archive.close()
	}
	for name in names {
		member_name := archive_member_name(name)!
		archive.open_entry(member_name)!
		payload := npy.to_bytes[T](arrays[name])!
		archive.write_entry(payload)!
		archive.close_entry()
	}
}

// write_arrays stores encoded .npy arrays of different element types in one
// NumPy .npz archive. Build each value with array[T](tensor).
pub fn write_arrays(path string, arrays map[string]NpyArray) ! {
	names := sorted_member_names(arrays.keys())!
	mut archive := szip.open(path, .default_compression, .write)!
	defer {
		archive.close()
	}
	for name in names {
		member_name := archive_member_name(name)!
		archive.open_entry(member_name)!
		payload := arrays[name].to_npy()!
		archive.write_entry(payload)!
		archive.close_entry()
	}
}

// read loads one named tensor from a NumPy .npz archive. It reads the member
// bytes directly and never extracts archive paths to the filesystem.
pub fn read[T](path string, name string) !&vtl.Tensor[T] {
	validate_member_name(name)!
	member_name := if name.ends_with('.npy') { name } else { '${name}.npy' }
	mut archive := szip.open(path, .default_compression, .read_only)!
	defer {
		archive.close()
	}
	entry_count := archive.total()!
	for index in 0 .. entry_count {
		archive.open_entry_by_index(index)!
		if archive.name() != member_name {
			archive.close_entry()
			continue
		}
		entry_size := archive.size()
		if entry_size > u64(max_int) {
			return error('npz.read: member is too large for this platform')
		}
		mut payload := []u8{len: int(entry_size)}
		read_size := archive.read_entry_buf(payload.data, payload.len)!
		archive.close_entry()
		if read_size != payload.len {
			return error('npz.read: incomplete data for ${member_name}')
		}
		return npy.read_bytes[T](payload)
	}
	return error('npz.read: archive does not contain ${member_name}')
}

fn validate_member_name(name string) ! {
	if name.len == 0 || name == '.' || name == '..' || name.contains('/') || name.contains('\\') {
		return error('npz: member name must be a non-empty filename without path separators')
	}
}

fn archive_member_name(name string) !string {
	validate_member_name(name)!
	return if name.ends_with('.npy') { name } else { '${name}.npy' }
}

fn sorted_member_names(keys []string) ![]string {
	mut names := keys.clone()
	names.sort()
	mut seen := map[string]bool{}
	for name in names {
		member_name := archive_member_name(name)!
		if member_name in seen {
			return error('npz: multiple names resolve to ${member_name}')
		}
		seen[member_name] = true
	}
	return names
}

// members lists the names of .npy arrays in a NumPy .npz archive without the
// filename suffix. Non-NPY ZIP entries are ignored.
pub fn members(path string) ![]string {
	mut archive := szip.open(path, .default_compression, .read_only)!
	defer {
		archive.close()
	}
	entry_count := archive.total()!
	mut result := []string{cap: entry_count}
	for index in 0 .. entry_count {
		archive.open_entry_by_index(index)!
		name := archive.name().clone()
		archive.close_entry()
		if name.ends_with('.npy') {
			result << name[..name.len - 4]
		}
	}
	return result
}
