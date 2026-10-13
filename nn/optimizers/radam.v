module optimizers

import math
import vtl
import vtl.autograd
import vtl.nn.types

// RAdamOptimizer implements Rectified Adam with adaptive variance rectification.
// Rectification threshold and epsilon placement follow the PyTorch RAdam algorithm.
pub struct RAdamOptimizer[T] {
	learning_rate          f64
	epsilon                f64
	weight_decay           f64
	decoupled_weight_decay bool
pub mut:
	beta1          f64
	beta2          f64
	beta1_t        f64
	beta2_t        f64
	step           int
	params         []&autograd.Variable[T]
	first_moments  []&vtl.Tensor[T]
	second_moments []&vtl.Tensor[T]
	groups         []&RAdamParamGroup[T]
}

// RAdamOptimizerConfig configures RAdamOptimizer.
@[params]
pub struct RAdamOptimizerConfig {
pub:
	learning_rate          f64 = 0.001
	beta1                  f64 = 0.9
	beta2                  f64 = 0.999
	epsilon                f64 = 1e-8
	weight_decay           f64
	decoupled_weight_decay bool
}

// radam_optimizer creates a RAdam optimizer.
pub fn radam_optimizer[T](config RAdamOptimizerConfig) &RAdamOptimizer[T] {
	return &RAdamOptimizer[T]{
		learning_rate:          config.learning_rate
		beta1:                  config.beta1
		beta2:                  config.beta2
		epsilon:                config.epsilon
		weight_decay:           config.weight_decay
		decoupled_weight_decay: config.decoupled_weight_decay
		beta1_t:                1.0
		beta2_t:                1.0
	}
}

// checkpoint_config returns the scalar RAdam state needed to resume training.
pub fn (o &RAdamOptimizer[T]) checkpoint_config() map[string]f64 {
	return {
		'learning_rate':          o.learning_rate
		'beta1':                  o.beta1
		'beta2':                  o.beta2
		'beta1_t':                o.beta1_t
		'beta2_t':                o.beta2_t
		'step':                   f64(o.step)
		'epsilon':                o.epsilon
		'weight_decay':           o.weight_decay
		'decoupled_weight_decay': if o.decoupled_weight_decay { 1.0 } else { 0.0 }
	}
}

// restore_checkpoint_state restores RAdam scalars and moment buffers.
pub fn (mut o RAdamOptimizer[T]) restore_checkpoint_state(config map[string]f64,
	first_moments []&vtl.Tensor[T], second_moments []&vtl.Tensor[T]) ! {
	o.validate_checkpoint_state(config, first_moments, second_moments)!
	o.learning_rate = config['learning_rate']
	o.beta1 = config['beta1']
	o.beta2 = config['beta2']
	o.beta1_t = config['beta1_t']
	o.beta2_t = config['beta2_t']
	o.step = int(config['step'])
	o.epsilon = config['epsilon']
	o.weight_decay = config['weight_decay']
	o.decoupled_weight_decay = config['decoupled_weight_decay'] == 1.0
	o.first_moments = first_moments.clone()
	o.second_moments = second_moments.clone()
}

// validate_checkpoint_state checks RAdam scalars and moments without mutation.
pub fn (o &RAdamOptimizer[T]) validate_checkpoint_state(config map[string]f64,
	first_moments []&vtl.Tensor[T], second_moments []&vtl.Tensor[T]) ! {
	for key in ['learning_rate', 'beta1', 'beta2', 'beta1_t', 'beta2_t', 'step', 'epsilon',
		'weight_decay', 'decoupled_weight_decay'] {
		if key !in config || !math.is_finite(config[key]) {
			return error('RAdamOptimizer.restore_checkpoint_state: missing or non-finite ${key}')
		}
	}
	lr := config['learning_rate']
	beta1 := config['beta1']
	beta2 := config['beta2']
	beta1_t := config['beta1_t']
	beta2_t := config['beta2_t']
	step := config['step']
	epsilon := config['epsilon']
	weight_decay := config['weight_decay']
	decoupled := config['decoupled_weight_decay']
	if lr < 0 || beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || beta1_t < 0
		|| beta1_t > 1 || beta2_t < 0 || beta2_t > 1 || step < 0
		|| step > 9007199254740992.0 || math.floor(step) != step || epsilon <= 0
		|| weight_decay < 0 || (decoupled != 0.0 && decoupled != 1.0) {
		return error('RAdamOptimizer.restore_checkpoint_state: scalar values are outside valid ranges')
	}
	if o.params.len != first_moments.len || o.params.len != second_moments.len {
		return error('RAdamOptimizer.restore_checkpoint_state: parameter and moment counts do not match')
	}
	for i, parameter in o.params {
		if first_moments[i].shape != parameter.value.shape
			|| second_moments[i].shape != parameter.value.shape {
			return error('RAdamOptimizer.restore_checkpoint_state: moment shape mismatch at parameter ${i}')
		}
	}
}

// build_params registers trainable variables and initializes moment estimates.
pub fn (mut o RAdamOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.first_moments << vtl.zeros_like[T](v.grad)
			o.second_moments << vtl.zeros_like[T](v.grad)
		}
	}
}

// RAdamParamGroup is a set of parameters sharing one RAdam configuration,
// mirroring PyTorch parameter groups, with its own step counter and moment
// buffers.
pub struct RAdamParamGroup[T] {
pub mut:
	params                 []&autograd.Variable[T]
	learning_rate          f64
	beta1                  f64
	beta2                  f64
	epsilon                f64
	weight_decay           f64
	decoupled_weight_decay bool
mut:
	beta1_t        f64
	beta2_t        f64
	step           int
	first_moments  []&vtl.Tensor[T]
	second_moments []&vtl.Tensor[T]
}

// add_param_group registers parameters under their own RAdam configuration
// and returns the group index. Register each variable either flat (via
// build_params) or in exactly one group.
pub fn (mut o RAdamOptimizer[T]) add_param_group(params []&autograd.Variable[T], config RAdamOptimizerConfig) !int {
	if config.learning_rate < 0 || config.beta1 < 0 || config.beta1 >= 1 || config.beta2 < 0
		|| config.beta2 >= 1 || config.epsilon <= 0 || config.weight_decay < 0 {
		return error('RAdam group learning rate and weight decay must be non-negative, betas must lie in [0, 1), and epsilon must be positive')
	}
	mut group := &RAdamParamGroup[T]{
		learning_rate:          config.learning_rate
		beta1:                  config.beta1
		beta2:                  config.beta2
		epsilon:                config.epsilon
		weight_decay:           config.weight_decay
		decoupled_weight_decay: config.decoupled_weight_decay
		beta1_t:                1.0
		beta2_t:                1.0
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
pub fn (o &RAdamOptimizer[T]) group_count() int {
	return o.groups.len
}

// group_learning_rate returns the current learning rate of one group.
pub fn (o &RAdamOptimizer[T]) group_learning_rate(group_index int) !f64 {
	if group_index < 0 || group_index >= o.groups.len {
		return error('RAdam group index ${group_index} out of range for ${o.groups.len} groups')
	}
	return o.groups[group_index].learning_rate
}

// set_group_learning_rate applies a new learning rate to one group only;
// this is the integration point for learning-rate schedulers.
pub fn (mut o RAdamOptimizer[T]) set_group_learning_rate(group_index int, learning_rate f64) ! {
	if learning_rate < 0 {
		return error('RAdam group learning rate must be non-negative')
	}
	if group_index < 0 || group_index >= o.groups.len {
		return error('RAdam group index ${group_index} out of range for ${o.groups.len} groups')
	}
	o.groups[group_index].learning_rate = learning_rate
}

fn (mut g RAdamParamGroup[T]) update_group() ! {
	g.step++
	g.beta1_t *= g.beta1
	g.beta2_t *= g.beta2
	step := RAdamStepParams{
		beta1:           g.beta1
		beta2:           g.beta2
		beta1_t:         g.beta1_t
		beta2_t:         g.beta2_t
		step:            g.step
		lr:              g.learning_rate
		epsilon:         g.epsilon
		weight_decay:    g.weight_decay
		decoupled_decay: g.decoupled_weight_decay
	}
	for i, mut parameter in g.params {
		if parameter.requires_grad {
			$if sizeof(T) == 8 {
				grad := parameter.grad.to_array()
				mut theta := parameter.value.to_array()
				mut m := g.first_moments[i].to_array()
				mut v := g.second_moments[i].to_array()
				radam_step_f64_cpu(grad, mut theta, mut m, mut v, step)
				parameter.value = vtl.from_array(theta, parameter.value.shape) or { return err }
				g.first_moments[i] = vtl.from_array(m, parameter.value.shape) or { return err }
				g.second_moments[i] = vtl.from_array(v, parameter.value.shape) or { return err }
			} $else $if sizeof(T) == 4 {
				grad := parameter.grad.to_array()
				mut theta := parameter.value.to_array()
				mut m := g.first_moments[i].to_array()
				mut v := g.second_moments[i].to_array()
				radam_step_f32_cpu(grad, mut theta, mut m, mut v, step)
				parameter.value = vtl.from_array(theta, parameter.value.shape) or { return err }
				g.first_moments[i] = vtl.from_array(m, parameter.value.shape) or { return err }
				g.second_moments[i] = vtl.from_array(v, parameter.value.shape) or { return err }
			} $else {
				return error('RAdamParamGroup.update: unsupported element type size ${sizeof(T)}')
			}
			parameter.grad = vtl.zeros_like[T](parameter.value)
		}
	}
}

// update performs one RAdam parameter update and zeros gradients.
pub fn (mut o RAdamOptimizer[T]) update() ! {
	for mut group in o.groups {
		group.update_group()!
	}
	o.step++
	o.beta1_t *= o.beta1
	o.beta2_t *= o.beta2
	step := RAdamStepParams{
		beta1:           o.beta1
		beta2:           o.beta2
		beta1_t:         o.beta1_t
		beta2_t:         o.beta2_t
		step:            o.step
		lr:              o.learning_rate
		epsilon:         o.epsilon
		weight_decay:    o.weight_decay
		decoupled_decay: o.decoupled_weight_decay
	}
	for i, mut parameter in o.params {
		if parameter.requires_grad {
			$if sizeof(T) == 8 {
				grad := parameter.grad.to_array()
				mut theta := parameter.value.to_array()
				mut m := o.first_moments[i].to_array()
				mut v := o.second_moments[i].to_array()
				radam_step_f64_cpu(grad, mut theta, mut m, mut v, step)
				parameter.value = vtl.from_array(theta, parameter.value.shape) or { return err }
				o.first_moments[i] = vtl.from_array(m, parameter.value.shape) or { return err }
				o.second_moments[i] = vtl.from_array(v, parameter.value.shape) or { return err }
			} $else $if sizeof(T) == 4 {
				grad := parameter.grad.to_array()
				mut theta := parameter.value.to_array()
				mut m := o.first_moments[i].to_array()
				mut v := o.second_moments[i].to_array()
				radam_step_f32_cpu(grad, mut theta, mut m, mut v, step)
				parameter.value = vtl.from_array(theta, parameter.value.shape) or { return err }
				o.first_moments[i] = vtl.from_array(m, parameter.value.shape) or { return err }
				o.second_moments[i] = vtl.from_array(v, parameter.value.shape) or { return err }
			} $else {
				return error('RAdamOptimizer.update: unsupported element type size ${sizeof(T)}')
			}
			parameter.grad = vtl.zeros_like[T](parameter.value)
		}
	}
}
