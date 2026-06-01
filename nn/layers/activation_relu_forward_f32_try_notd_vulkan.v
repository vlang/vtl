module layers

import vtl

pub fn relu_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	_ = x
	return none
}

pub fn sigmoid_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	_ = x
	return none
}
