module vtl

import rand
import time
import vtl.etype
import vtl.storage

fn rand_between(min etype.Num, max etype.Num) etype.Num {
	return etype.Num(rand.f64_in_range(etype.num_as_type<f64>(min), etype.num_as_type<f64>(max)))
}

pub fn random(min etype.Num, max etype.Num, shape []int) Tensor {
	ret := empty(shape)
	mut iter := ret.iterator()
	for _ in iter {
		r := rand_between(min, max)
		storage.storage_set(ret.data, iter.pos, r)
	}
	return ret
}

pub fn random_seed(i int) {
	rand.seed([u32(i), u32(i)])
}

fn init() {
	rand.seed([u32(time.now().unix), u32(time.now().unix)])
}
