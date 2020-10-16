module num

import rand
import time

fn rand_between(min f64, max f64) f64 {
	return rand.f64_in_range(min, max)
}

pub fn random(min f64, max f64, shape []int) NdArray {
	ret := empty(shape)
	for i := ret.iter(); !i.done; i.next() {
		unsafe {
			*i.ptr = rand_between(min, max)
		}
	}
	return ret
}

pub fn random_seed(i int) {
	rand.seed([u32(i), u32(i)])
}

fn init() {
	rand.seed([u32(time.now().unix), u32(time.now().unix)])
}
