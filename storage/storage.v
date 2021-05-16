module storage

import vtl.etype

pub enum StorageStrategy {
	cpu
}

pub struct NullStorage {}

// sum type to be used for different strategies
// pub interface Storage {
// 	element_size int
// 	data voidptr
// 	len int
// 	capacity int
// 	get(i int) voidptr
// 	set(i int, val voidptr)
// 	fill(val voidptr)
// 	clone() Storage
// 	slice(start int, end int) Storage
// 	offset(start int) Storage
// }
pub type Storage = CpuStorage | NullStorage

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
	match s {
		CpuStorage {
			return create_storage(s.len, s.capacity, s.element_size, voidptr(0), storage_strategy(s))
		}
		else {
			panic('unsupported storage')
		}
	}
}

[inline]
pub fn new_storage_like_with_len(s Storage, len int) Storage {
	match s {
		CpuStorage {
			return create_storage(len, 0, s.element_size, voidptr(0), storage_strategy(s))
		}
		else {
			panic('unsupported storage')
		}
	}
}

pub fn create_storage(len int, cap int, element_size int, init voidptr, strategy StorageStrategy) Storage {
	if strategy == .cpu {
		return new_cpu_with_default(len, cap, element_size, init)
	}
	return new_cpu_with_default(len, cap, element_size, init)
}

pub fn create_storage_from_c_array(len int, cap int, element_size int, c_array voidptr, strategy StorageStrategy) Storage {
	if strategy == .cpu {
		return unsafe { new_cpu_from_c_array(len, cap, element_size, c_array) }
	}
	return unsafe { new_cpu_from_c_array(len, cap, element_size, c_array) }
}

pub fn storage_to_varray<T>(s Storage) []T {
	match s {
		CpuStorage {
			if s.element_size == int(sizeof(T)) {
				mut arr := []T{}
				unsafe { arr.push_many(s.data, s.len) }
				return arr
			}
			panic('CpuStorage.to_varray: incoming type T does not match with the stored data type')
		}
		else {
			panic('storage not allowed')
		}
	}
}

pub fn storage_strategy(s Storage) StorageStrategy {
	match s {
		CpuStorage { return .cpu }
		else { panic('storage not allowed') }
	}
}

pub fn storage_clone(s Storage) Storage {
	match s {
		CpuStorage { return unsafe { s.clone() } }
		else { panic('unsupported storage') }
	}
}

pub fn storage_offset(s Storage, start int) Storage {
	match s {
		CpuStorage { return unsafe { s.offset(start) } }
		else { panic('unsupported storage') }
	}
}

pub fn storage_get(s Storage, i int, element_type string) etype.Num {
	match s {
		CpuStorage { return etype.ptr_to_val_of_type(unsafe { s.get(i) }, element_type) }
		else { panic('unsupported storage') }
	}
}

pub fn storage_set(s Storage, i int, val etype.Num) {
	match s {
		CpuStorage { unsafe { s.set(i, val.ptr()) } }
		else { panic('unsupported storage') }
	}
}

pub fn storage_fill(s Storage, val etype.Num) {
	match s {
		CpuStorage { unsafe { s.fill(val.ptr()) } }
		else { panic('unsupported storage') }
	}
}

pub fn storage_size(s Storage) int {
	match s {
		CpuStorage { return s.len }
		else { panic('unsupported storage') }
	}
}
