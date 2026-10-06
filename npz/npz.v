module npz

import compress.szip
import vtl
import vtl.npy

// write stores one or more same-element-type tensors in a NumPy .npz archive.
// The archive member name is the map key with a .npy suffix appended when it
// is not already present.
pub fn write[T](path string, arrays map[string]&vtl.Tensor[T]) ! {
	mut archive := szip.open(path, .default_compression, .write)!
	defer {
		archive.close()
	}
	mut names := arrays.keys()
	names.sort()
	mut previous_member_name := ''
	for name in names {
		validate_member_name(name)!
		member_name := if name.ends_with('.npy') { name } else { '${name}.npy' }
		if member_name == previous_member_name {
			return error('npz.write: multiple names resolve to ${member_name}')
		}
		archive.open_entry(member_name)!
		payload := npy.to_bytes[T](arrays[name])!
		archive.write_entry(payload)!
		archive.close_entry()
		previous_member_name = member_name
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
