module vtl

import vtl.storage

pub fn new_storage_from_varray<T>(arr []T, strategy storage.StorageStrategy) storage.Storage {
	return storage.create_storage_from_c_array(arr.len, 0, arr.element_size, arr.data,
		strategy)
}

pub fn storage_to_varray<T>(s storage.Storage) []T {
	match s {
		storage.CpuStorage {
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
