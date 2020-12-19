module vtl

import rand
import time

fn rand_between(min Num, max Num) Num {
	return Num(rand.f64_in_range(num_as_type<f64>(min), num_as_type<f64>(max)))
}

pub fn random(min Num, max Num, shape []int) Tensor {
	ret := empty(shape)
	mut iter := ret.iterator()
	for _ in 0 .. ret.size {
		r := rand_between(min, max)
		storage_set(ret.data, iter.pos, r)
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
