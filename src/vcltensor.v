module vtl

import vtl.storage

// VclTensor is the main structure defined by VTL to manage N Dimensional data
[heap]
pub struct VclTensor<T> {
pub mut:
	data    &storage.VclStorage<T>
	memory  MemoryFormat
	size    int
	shape   []int
	strides []int
}

// vcl
pub fn (t &Tensor<T>) vcl(params storage.VclStorageParams) ?&VclTensor<T> {
	cldata := t.data.vcl(params)?
	return &VclTensor<T>{
		data: cldata
	}
}
