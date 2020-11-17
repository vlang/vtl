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
	init     voidptr
	strategy StorageStrategy = .cpu
}

[inline]
pub fn new_storage<T>(data StorageData) Storage {
	element_size := int(sizeof(T))
	return create_storage(data.len, data.cap, element_size, data.init, data.strategy)
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
pub fn new_storage_from_varray<T>(arr []T, strategy StorageStrategy) Storage {
	return create_storage_from_c_array(arr.len, 0, int(sizeof(T)), arr.data, strategy)
}

pub fn storage_to_varray<T>(s Storage) []T {
	match s {
		storage.CpuStorage {
			if s.element_size == int(sizeof(T)) {
				mut arr := []T{}
				arr.push_many(s.data, s.len)
				return arr
			}
			panic('CpuStorage.to_varray<T>: incoming type T does not match with the stored data type')
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
		storage.CpuStorage { return s.clone() }
		else { panic('storage not allowed') }
	}
}

fn storage_get(s Storage, i int) voidptr {
	match s {
		storage.CpuStorage { return s.get(i) }
		else { panic('storage not allowed') }
	}
}

fn storage_set(s Storage, i int, val voidptr) {
	match mut s {
		storage.CpuStorage { s.set(i, val) }
		else { panic('storage not allowed') }
	}
}

fn storage_fill(s Storage, val voidptr) {
	match mut s {
		storage.CpuStorage { s.fill(val) }
		else { panic('storage not allowed') }
	}
}
