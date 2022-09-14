module nn

import vtl.autograd
import vtl.nn.layers

pub struct NeuralNetwork<T> {
	info &NeuralNetworkContainer<T>
}

pub fn new_nn<T>(ctx &autograd.Context<T>) &NeuralNetwork<T> {
	return &NeuralNetwork<T>{
		info: new_nnc<T>(ctx)
	}
}

pub fn (mut nn NeuralNetwork<T>) forward(train &autograd.Variable<T>) ?&autograd.Variable<T> {
	mut ret := unsafe { train }
	for layer in nn.info.layers {
		ret = layers.layer_forward<T>(layer, mut ret)?
	}
	return ret
}
