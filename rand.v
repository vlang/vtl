module vtl

import rand
import time

fn init() {
	rand.seed([u32(time.now().unix), u32(time.now().unix)])
}

pub fn random_seed(i int) {
	rand.seed([u32(i), u32(i)])
}
