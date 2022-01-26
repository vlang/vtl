module nn

import vtl.autograd
import vtl.nn.layers

pub struct NNContainer<T> {
	ctx &autograd.Context<T>
pub mut:
	layers []Layer
}

pub fn new_nn_container<T>(ctx &autograd.Context<T>) &NNContainer<T> {
	return &NNContainer<T>{
		ctx: ctx
	}
}

pub fn (mut nncon NNContainer<T>) input(shape []int) {
	nncon.layers << layers.new_input_layer(nncon.ctx, shape)
}
