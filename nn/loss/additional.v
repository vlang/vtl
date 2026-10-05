module loss

import vtl
import vtl.autograd
import vtl.nn.types
import vtl.nn.internal
import vtl.nn.gates.loss

// L1Loss computes mean absolute error.
pub struct L1Loss[T] {}

pub fn l1_loss[T]() types.Loss[T] {
	concrete := &L1Loss[T]{}
	return types.loss[T](voidptr(concrete), l1_loss_dispatch[T])
}

pub fn (_ &L1Loss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.l1[T](input.value, target)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		loss.additional_loss_gate[T](input.value, target, 'l1', 0, 0, false).cache(mut result, input)!
	}
	return result
}

fn l1_loss_dispatch[T](loss_ptr voidptr, input voidptr, target voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	typed_target := unsafe { &vtl.Tensor[T](target) }
	result := unsafe { (&L1Loss[T](loss_ptr)).loss(typed_input, typed_target)! }
	return voidptr(result)
}

// HingeLoss computes mean binary hinge loss for labels in {-1, +1}.
pub struct HingeLoss[T] {}

pub fn hinge_loss[T]() types.Loss[T] {
	concrete := &HingeLoss[T]{}
	return types.loss[T](voidptr(concrete), hinge_loss_dispatch[T])
}

pub fn (_ &HingeLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.hinge[T](input.value, target)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		loss.additional_loss_gate[T](input.value, target, 'hinge', 0, 0, false).cache(mut result, input)!
	}
	return result
}

fn hinge_loss_dispatch[T](loss_ptr voidptr, input voidptr, target voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	typed_target := unsafe { &vtl.Tensor[T](target) }
	result := unsafe { (&HingeLoss[T](loss_ptr)).loss(typed_input, typed_target)! }
	return voidptr(result)
}

// FocalLossConfig configures binary focal loss; alpha weights positive examples and gamma focuses hard examples.
@[params]
pub struct FocalLossConfig {
pub:
	alpha       f64  = 0.25
	gamma       f64  = 2.0
	from_logits bool = true
}

// FocalLoss computes mean binary focal loss.
pub struct FocalLoss[T] {
	config FocalLossConfig
}

pub fn focal_loss[T](config FocalLossConfig) types.Loss[T] {
	concrete := &FocalLoss[T]{ config: config }
	return types.loss[T](voidptr(concrete), focal_loss_dispatch[T])
}

pub fn (l &FocalLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.focal[T](input.value, target, l.config.alpha, l.config.gamma, l.config.from_logits)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		loss.additional_loss_gate[T](input.value, target, 'focal', l.config.alpha, l.config.gamma, l.config.from_logits).cache(mut result, input)!
	}
	return result
}

fn focal_loss_dispatch[T](loss_ptr voidptr, input voidptr, target voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	typed_target := unsafe { &vtl.Tensor[T](target) }
	result := unsafe { (&FocalLoss[T](loss_ptr)).loss(typed_input, typed_target)! }
	return voidptr(result)
}
