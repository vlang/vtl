module nn

import vtl.autograd

pub struct NeuralNetwork<T> {
	info &NeuralNetworkContainer<T>
}

pub fn new_nn<T>(ctx &autograd.Context<T>) &NeuralNetwork<T> {
	return &NeuralNetwork<T>{
		info: new_nnc<T>(ctx)
	}
}
