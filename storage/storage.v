module storage

import vtl.etype

pub enum StorageStrategy {
	cpu
}

pub interface Storage {
	element_size int
	data voidptr
	len int
	capacity int
	get(index int) voidptr
	set(index int, val voidptr)
	offset(start int) Storage
	fill(val voidptr)
	clone() Storage
}

[heap]
pub struct StorageData {
pub:
	len      int
	cap      int
	init     etype.Num       = etype.default_init
	etype    string          = etype.default_type
	strategy StorageStrategy = .cpu
}

[inline]
pub fn new_storage(data StorageData) Storage {
	element_size := etype.str_esize(data.etype)
	return create_storage(data.len, data.cap, element_size, data.init.ptr(), data.strategy)
}

[inline]
pub fn new_storage_like(s Storage) Storage {
	return create_storage(s.len, s.capacity, s.element_size, voidptr(0), storage_strategy(s))
}

[inline]
pub fn new_storage_like_with_len(s Storage, len int) Storage {
	return create_storage(len, 0, s.element_size, voidptr(0), storage_strategy(s))
}

[inline]
pub fn create_storage(len int, cap int, element_size int, init voidptr, strategy StorageStrategy) Storage {
	if strategy == .cpu {
		return new_cpu_with_default(len, cap, element_size, init)
	}
	return new_cpu_with_default(len, cap, element_size, init)
}

[inline]
pub fn create_storage_from_c_array(len int, cap int, element_size int, c_array voidptr, strategy StorageStrategy) Storage {
	if strategy == .cpu {
		return new_cpu_from_c_array(len, cap, element_size, c_array)
	}
	return new_cpu_from_c_array(len, cap, element_size, c_array)
}

[inline]
pub fn storage_to_varray<T>(s Storage) []T {
	match s {
		CpuStorage {
			if s.element_size == int(sizeof(T)) {
				mut arr := []T{}
				arr.push_many(s.data, s.len)
				return arr
			}
			panic('CpuStorage.to_varray: incoming type T does not match with the stored data type')
		}
		else {
			panic('storage not allowed')
		}
	}
}

[inline]
pub fn storage_strategy(s Storage) StorageStrategy {
	match s {
		CpuStorage { return .cpu }
		else { panic('storage not allowed') }
	}
}

[inline]
pub fn storage_get(s Storage, idx int, element_type string) etype.Num {
	corrected_idx := if idx < 0 { s.len + idx } else { idx }
	return etype.ptr_to_val_of_type(s.get(corrected_idx), element_type)
}

[inline]
pub fn storage_fill(s Storage, val etype.Num) {
	s.fill(val.ptr())
}

[inline]
pub fn storage_set(s Storage, idx int, val etype.Num) {
	corrected_idx := if idx < 0 { s.len + idx } else { idx }
	s.set(corrected_idx, val.ptr())
}
