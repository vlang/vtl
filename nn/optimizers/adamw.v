module optimizers

import math
import vtl.autograd
import vtl.nn.types
import vtl

// AdamWOptimizer implements AdamW (Adam with Decoupled Weight Decay).
//
// Identical to Adam but weight decay is applied directly to the parameters
// (not through the gradient), which typically gives better generalisation.
//
// Update rule (after bias correction):
//   θ = θ - lr · (m̂ / (√v̂ + ε) + weight_decay · θ)
//
// Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017).
pub struct AdamWOptimizer[T] {
	learning_rate f64
	epsilon       f64
pub mut:
	beta1          f64
	beta2          f64
	beta1_t        f64
	beta2_t        f64
	weight_decay   f64
	params         []&autograd.Variable[T]
	first_moments  []&vtl.Tensor[T]
	second_moments []&vtl.Tensor[T]
	groups         []&AdamWParamGroup[T]
}

// AdamWOptimizerConfig configures AdamWOptimizer.
//
// Fields:
//   - `learning_rate` — step size (default: 0.001)
//   - `beta1`         — first-moment decay rate (default: 0.9)
//   - `beta2`         — second-moment decay rate (default: 0.999)
//   - `epsilon`       — numerical stability constant (default: 1e-8)
//   - `weight_decay`  — decoupled weight-decay coefficient λ (default: 0.01)

// AdamWOptimizerConfig defines a public data structure for this module.

// AdamWOptimizerConfig defines a public data structure for this module.
@[params]
pub struct AdamWOptimizerConfig {
pub:
	learning_rate f64 = 0.001
	beta1         f64 = 0.9
	beta2         f64 = 0.999
	epsilon       f64 = 1e-8
	weight_decay  f64 = 0.01
}

// adamw creates a new AdamWOptimizer.
pub fn adamw[T](config AdamWOptimizerConfig) &AdamWOptimizer[T] {
	return &AdamWOptimizer[T]{
		learning_rate: config.learning_rate
		beta1:         config.beta1
		beta2:         config.beta2
		epsilon:       config.epsilon
		weight_decay:  config.weight_decay
		beta1_t:       config.beta1
		beta2_t:       config.beta2
	}
}

// checkpoint_config returns the scalar AdamW state needed to resume training.
pub fn (o &AdamWOptimizer[T]) checkpoint_config() map[string]f64 {
	return {
		'learning_rate': o.learning_rate
		'beta1':         o.beta1
		'beta2':         o.beta2
		'beta1_t':       o.beta1_t
		'beta2_t':       o.beta2_t
		'epsilon':       o.epsilon
		'weight_decay':  o.weight_decay
	}
}

// restore_checkpoint_state restores AdamW configuration and moment buffers.
// Parameters must already be registered with build_params in the same order.
pub fn (mut o AdamWOptimizer[T]) restore_checkpoint_state(config map[string]f64,
	first_moments []&vtl.Tensor[T], second_moments []&vtl.Tensor[T]) ! {
	o.validate_checkpoint_state(config, first_moments, second_moments)!
	o.learning_rate = config['learning_rate']
	o.beta1 = config['beta1']
	o.beta2 = config['beta2']
	o.beta1_t = config['beta1_t']
	o.beta2_t = config['beta2_t']
	o.epsilon = config['epsilon']
	o.weight_decay = config['weight_decay']
	o.first_moments = first_moments.clone()
	o.second_moments = second_moments.clone()
}

// validate_checkpoint_state checks AdamW scalars and moments without mutation.
pub fn (o &AdamWOptimizer[T]) validate_checkpoint_state(config map[string]f64,
	first_moments []&vtl.Tensor[T], second_moments []&vtl.Tensor[T]) ! {
	for key in ['learning_rate', 'beta1', 'beta2', 'beta1_t', 'beta2_t', 'epsilon', 'weight_decay'] {
		if key !in config || !math.is_finite(config[key]) {
			return error('AdamWOptimizer.restore_checkpoint_state: missing or non-finite ${key}')
		}
	}
	lr := config['learning_rate']
	beta1 := config['beta1']
	beta2 := config['beta2']
	beta1_t := config['beta1_t']
	beta2_t := config['beta2_t']
	epsilon := config['epsilon']
	weight_decay := config['weight_decay']
	if lr < 0 || beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || beta1_t < 0
		|| beta1_t >= 1 || beta2_t < 0 || beta2_t >= 1 || epsilon <= 0 || weight_decay < 0 {
		return error('AdamWOptimizer.restore_checkpoint_state: scalar values are outside valid ranges')
	}
	if o.params.len != first_moments.len || o.params.len != second_moments.len {
		return error('AdamWOptimizer.restore_checkpoint_state: parameter and moment counts do not match')
	}
	for i, parameter in o.params {
		if first_moments[i].shape != parameter.value.shape
			|| second_moments[i].shape != parameter.value.shape {
			return error('AdamWOptimizer.restore_checkpoint_state: moment shape mismatch at parameter ${i}')
		}
	}
}

// AdamWParamGroup is a set of parameters sharing one AdamW configuration,
// mirroring PyTorch parameter groups, with its own bias-correction counters
// and moment buffers.
pub struct AdamWParamGroup[T] {
pub mut:
	params        []&autograd.Variable[T]
	learning_rate f64
	beta1         f64
	beta2         f64
	epsilon       f64
	weight_decay  f64
mut:
	beta1_t        f64
	beta2_t        f64
	first_moments  []&vtl.Tensor[T]
	second_moments []&vtl.Tensor[T]
}

// add_param_group registers parameters under their own AdamW configuration
// and returns the group index. Register each variable either flat (via
// build_params) or in exactly one group.
pub fn (mut o AdamWOptimizer[T]) add_param_group(params []&autograd.Variable[T], config AdamWOptimizerConfig) !int {
	if config.learning_rate < 0 || config.beta1 < 0 || config.beta1 >= 1 || config.beta2 < 0
		|| config.beta2 >= 1 || config.epsilon <= 0 || config.weight_decay < 0 {
		return error('AdamW group learning rate and weight decay must be non-negative, betas must lie in [0, 1), and epsilon must be positive')
	}
	mut group := &AdamWParamGroup[T]{
		learning_rate: config.learning_rate
		beta1:         config.beta1
		beta2:         config.beta2
		epsilon:       config.epsilon
		weight_decay:  config.weight_decay
		beta1_t:       config.beta1
		beta2_t:       config.beta2
	}
	for v in params {
		group.params << v
		group.first_moments << vtl.zeros_like[T](v.grad)
		group.second_moments << vtl.zeros_like[T](v.grad)
	}
	o.groups << group
	return o.groups.len - 1
}

// group_count returns how many parameter groups are registered.
pub fn (o &AdamWOptimizer[T]) group_count() int {
	return o.groups.len
}

// group_learning_rate returns the current learning rate of one group.
pub fn (o &AdamWOptimizer[T]) group_learning_rate(group_index int) !f64 {
	if group_index < 0 || group_index >= o.groups.len {
		return error('AdamW group index ${group_index} out of range for ${o.groups.len} groups')
	}
	return o.groups[group_index].learning_rate
}

// set_group_learning_rate applies a new learning rate to one group only;
// this is the integration point for learning-rate schedulers.
pub fn (mut o AdamWOptimizer[T]) set_group_learning_rate(group_index int, learning_rate f64) ! {
	if learning_rate < 0 {
		return error('AdamW group learning rate must be non-negative')
	}
	if group_index < 0 || group_index >= o.groups.len {
		return error('AdamW group index ${group_index} out of range for ${o.groups.len} groups')
	}
	o.groups[group_index].learning_rate = learning_rate
}

fn (mut g AdamWParamGroup[T]) update_group() ! {
	lr_t := g.learning_rate * math.sqrt(1.0 - g.beta2_t) / (1.0 - g.beta1_t)
	g.beta1_t *= g.beta1
	g.beta2_t *= g.beta2
	for i, mut v in g.params {
		if v.requires_grad {
			// napply passes the moment tensor itself as vals[0] and the
			// gradient as vals[1]: m = beta1*m + (1-beta1)*grad.
			g.first_moments[i].napply([v.grad], fn [g] [T](vals []T, idx []int) T {
				return vtl.cast[T](g.beta1 * f64(vals[0]) + (1.0 - g.beta1) * f64(vals[1]))
			}) or { return err }

			g.second_moments[i].napply([v.grad], fn [g] [T](vals []T, idx []int) T {
				grad := f64(vals[1])
				return vtl.cast[T](g.beta2 * f64(vals[0]) + (1.0 - g.beta2) * grad * grad)
			}) or { return err }

			v.value.napply([g.first_moments[i], g.second_moments[i]], fn [g, lr_t] [T](vals []T, idx []int) T {
				theta := f64(vals[0])
				m := f64(vals[1])
				v_ := f64(vals[2])
				wd := g.weight_decay
				return vtl.cast[T](theta - lr_t * (m / (math.sqrt(v_) + g.epsilon) + wd * theta))
			}) or { return err }

			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}

// build_params registers all trainable variables from `layers`. Call once before training.
pub fn (mut o AdamWOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.first_moments << vtl.zeros_like[T](v.grad)
			o.second_moments << vtl.zeros_like[T](v.grad)
		}
	}
}

// update performs one AdamW parameter update and zeros all gradients.
pub fn (mut o AdamWOptimizer[T]) update() ! {
	for mut group in o.groups {
		group.update_group()!
	}
	lr_t := o.learning_rate * math.sqrt(1.0 - o.beta2_t) / (1.0 - o.beta1_t)

	o.beta1_t *= o.beta1
	o.beta2_t *= o.beta2

	for i, mut v in o.params {
		if v.requires_grad {
			// napply passes the moment tensor itself as vals[0] and the
			// gradient as vals[1]: m = beta1*m + (1-beta1)*grad.
			o.first_moments[i].napply([v.grad], fn [o] [T](vals []T, idx []int) T {
				return vtl.cast[T](o.beta1 * f64(vals[0]) + (1.0 - o.beta1) * f64(vals[1]))
			}) or { return err }

			o.second_moments[i].napply([v.grad], fn [o] [T](vals []T, idx []int) T {
				grad := f64(vals[1])
				return vtl.cast[T](o.beta2 * f64(vals[0]) + (1.0 - o.beta2) * grad * grad)
			}) or { return err }

			v.value.napply([o.first_moments[i], o.second_moments[i]], fn [o, lr_t] [T](vals []T, idx []int) T {
				theta := f64(vals[0])
				m := f64(vals[1])
				v_ := f64(vals[2])
				wd := o.weight_decay
				return vtl.cast[T](theta - lr_t * (m / (math.sqrt(v_) + o.epsilon) + wd * theta))
			}) or { return err }

			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}
