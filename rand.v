module vtl

import rand
import time

fn rand_between<T>(min T, max T) T {
	return T(rand.f64_in_range(min, max))
}

pub fn random<T>(min T, max T, shape []int) Tensor {
	ret := empty<T>(shape)
	mut iter := ret.iterator()
	for _ in 0 .. ret.size {
		r := rand_between<T>(min, max)
		storage_set(ret.data, iter.pos, &r)
		iter.next()
	}
	return ret
}

pub fn random_seed(i int) {
	rand.seed([u32(i), u32(i)])
}

fn init() {
	rand.seed([u32(time.now().unix), u32(time.now().unix)])
}
