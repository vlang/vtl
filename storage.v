module vtl

import vtl.storage

pub enum StorageStrategy {
	cpu
}

pub struct NullStorage {
}

// sum type to be used for different strategies
pub type Storage = NullStorage | storage.CpuStorage

pub struct StorageData {
pub:
	len      int
	cap      int
	init     Num    = default_init
	etype    string = default_type
	strategy StorageStrategy = .cpu
}

[inline]
pub fn new_storage(data StorageData) Storage {
	element_size := str_esize(data.etype)
	return create_storage(data.len, data.cap, element_size, data.init.ptr(), data.strategy)
}

[inline]
pub fn new_storage_like(s Storage) Storage {
	match s {
		storage.CpuStorage { return create_storage(s.len, s.capacity, s.element_size,
				voidptr(0), storage_strategy(s)) }
		else { panic('storage not allowed') }
	}
}

[inline]
pub fn new_storage_like_with_len(s Storage, len int) Storage {
	match s {
		storage.CpuStorage { return create_storage(len, 0, s.element_size, voidptr(0),
				storage_strategy(s)) }
		else { panic('storage not allowed') }
	}
}

[inline]
pub fn new_storage_from_varray<T>(arr []T, strategy StorageStrategy) Storage {
	return create_storage_from_c_array(arr.len, 0, arr.element_size, arr.data, strategy)
}

pub fn storage_to_varray<T>(s Storage) []T {
	match s {
		storage.CpuStorage {
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

fn create_storage(len int, cap int, element_size int, init voidptr, strategy StorageStrategy) Storage {
	if strategy == .cpu {
		return storage.new_cpu_with_default(len, cap, element_size, init)
	}
	return storage.new_cpu_with_default(len, cap, element_size, init)
}

fn create_storage_from_c_array(len int, cap int, element_size int, c_array voidptr, strategy StorageStrategy) Storage {
	if strategy == .cpu {
		return storage.new_cpu_from_c_array(len, cap, element_size, c_array)
	}
	return storage.new_cpu_from_c_array(len, cap, element_size, c_array)
}

fn storage_strategy(s Storage) StorageStrategy {
	match s {
		storage.CpuStorage { return .cpu }
		else { panic('storage not allowed') }
	}
}

fn storage_clone(s Storage) Storage {
	match s {
		storage.CpuStorage { return unsafe { s.clone() } }
		else { panic('storage not allowed') }
	}
}

fn storage_offset(s Storage, start int) Storage {
	match s {
		storage.CpuStorage { return unsafe { s.offset(start) } }
		else { panic('storage not allowed') }
	}
}

fn storage_get(s Storage, i int, etype string) Num {
	match s {
		storage.CpuStorage { return ptr_to_val_of_type(unsafe { s.get(i) }, etype) }
		else { panic('storage not allowed') }
	}
}

fn storage_set(s Storage, i int, val Num) {
	match mut s {
		storage.CpuStorage { unsafe { s.set(i, val.ptr()) } }
		else { panic('storage not allowed') }
	}
}

fn storage_fill(s Storage, val Num) {
	match mut s {
		storage.CpuStorage { unsafe { s.fill(val.ptr()) } }
		else { panic('storage not allowed') }
	}
}

fn storage_size(s Storage) int {
	match s {
		storage.CpuStorage { return s.len }
		else { panic('storage not allowed') }
	}
}
