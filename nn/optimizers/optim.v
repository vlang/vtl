module optimizers

import vtl.nn.types

pub fn optimize[T](mut optim types.Optimizer) ! {
	match mut optim {
		SgdOptimizer[T] {
			return optim.update()
		}
		AdamOptimizer[T] {
			return optim.update()
		}
		else {
			panic(@FN + ': Unknown layer type ${typeof(optim).name}')
		}
	}
}
