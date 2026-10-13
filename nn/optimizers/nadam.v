module optimizers

import math
import vtl
import vtl.autograd
import vtl.nn.types

// NAdamOptimizer implements Nesterov-accelerated Adam for f32 and f64 parameters.
// Reference: Dozat (2016), Incorporating Nesterov Momentum into Adam; PyTorch NAdam.
pub struct NAdamOptimizer[T] {
	learning_rate          f64
	epsilon                f64
	weight_decay           f64
	momentum_decay         f64
	decoupled_weight_decay bool
pub mut:
	beta1          f64
	beta2          f64
	beta1_t        f64
	beta2_t        f64
	mu_product_t   f64
	step           int
	params         []&autograd.Variable[T]
	first_moments  []&vtl.Tensor[T]
	second_moments []&vtl.Tensor[T]
}

// NAdamOptimizerConfig configures NAdamOptimizer.
@[params]
pub struct NAdamOptimizerConfig {
pub:
	learning_rate          f64 = 0.002
	beta1                  f64 = 0.9
	beta2                  f64 = 0.999
	epsilon                f64 = 1e-8
	weight_decay           f64
	momentum_decay         f64 = 0.004
	decoupled_weight_decay bool
}

// nadam_optimizer creates a NAdam optimizer.
pub fn nadam_optimizer[T](config NAdamOptimizerConfig) &NAdamOptimizer[T] {
	return &NAdamOptimizer[T]{
		learning_rate:          config.learning_rate
		beta1:                  config.beta1
		beta2:                  config.beta2
		epsilon:                config.epsilon
		weight_decay:           config.weight_decay
		momentum_decay:         config.momentum_decay
		decoupled_weight_decay: config.decoupled_weight_decay
		beta1_t:                1.0
		beta2_t:                1.0
		mu_product_t:           1.0
	}
}

// checkpoint_config returns the scalar NAdam state needed to resume training.
pub fn (o &NAdamOptimizer[T]) checkpoint_config() map[string]f64 {
	return {
		'learning_rate':          o.learning_rate
		'beta1':                  o.beta1
		'beta2':                  o.beta2
		'beta1_t':                o.beta1_t
		'beta2_t':                o.beta2_t
		'mu_product_t':           o.mu_product_t
		'step':                   f64(o.step)
		'epsilon':                o.epsilon
		'weight_decay':           o.weight_decay
		'momentum_decay':         o.momentum_decay
		'decoupled_weight_decay': if o.decoupled_weight_decay { 1.0 } else { 0.0 }
	}
}

// restore_checkpoint_state restores NAdam scalars and moment buffers.
pub fn (mut o NAdamOptimizer[T]) restore_checkpoint_state(config map[string]f64,
	first_moments []&vtl.Tensor[T], second_moments []&vtl.Tensor[T]) ! {
	o.validate_checkpoint_state(config, first_moments, second_moments)!
	o.learning_rate = config['learning_rate']
	o.beta1 = config['beta1']
	o.beta2 = config['beta2']
	o.beta1_t = config['beta1_t']
	o.beta2_t = config['beta2_t']
	o.mu_product_t = config['mu_product_t']
	o.step = int(config['step'])
	o.epsilon = config['epsilon']
	o.weight_decay = config['weight_decay']
	o.momentum_decay = config['momentum_decay']
	o.decoupled_weight_decay = config['decoupled_weight_decay'] == 1.0
	o.first_moments = first_moments.clone()
	o.second_moments = second_moments.clone()
}

// validate_checkpoint_state checks NAdam scalars and moments without mutation.
pub fn (o &NAdamOptimizer[T]) validate_checkpoint_state(config map[string]f64,
	first_moments []&vtl.Tensor[T], second_moments []&vtl.Tensor[T]) ! {
	for key in ['learning_rate', 'beta1', 'beta2', 'beta1_t', 'beta2_t', 'mu_product_t', 'step',
		'epsilon', 'weight_decay', 'momentum_decay', 'decoupled_weight_decay'] {
		if key !in config || !math.is_finite(config[key]) {
			return error('NAdamOptimizer.restore_checkpoint_state: missing or non-finite ${key}')
		}
	}
	lr := config['learning_rate']
	beta1 := config['beta1']
	beta2 := config['beta2']
	beta1_t := config['beta1_t']
	beta2_t := config['beta2_t']
	mu_product_t := config['mu_product_t']
	step := config['step']
	epsilon := config['epsilon']
	weight_decay := config['weight_decay']
	momentum_decay := config['momentum_decay']
	decoupled := config['decoupled_weight_decay']
	if lr < 0 || beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || beta1_t < 0
		|| beta1_t > 1 || beta2_t < 0 || beta2_t > 1 || mu_product_t < 0 || mu_product_t > 1
		|| step < 0 || step > 9007199254740992.0 || math.floor(step) != step || epsilon <= 0
		|| weight_decay < 0 || momentum_decay < 0 || (decoupled != 0.0 && decoupled != 1.0) {
		return error('NAdamOptimizer.restore_checkpoint_state: scalar values are outside valid ranges')
	}
	if o.params.len != first_moments.len || o.params.len != second_moments.len {
		return error('NAdamOptimizer.restore_checkpoint_state: parameter and moment counts do not match')
	}
	for i, parameter in o.params {
		if first_moments[i].shape != parameter.value.shape
			|| second_moments[i].shape != parameter.value.shape {
			return error('NAdamOptimizer.restore_checkpoint_state: moment shape mismatch at parameter ${i}')
		}
	}
}

// build_params registers trainable variables and initializes moment estimates.
pub fn (mut o NAdamOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.first_moments << vtl.zeros_like[T](v.grad)
			o.second_moments << vtl.zeros_like[T](v.grad)
		}
	}
}

// update performs one NAdam parameter update and zeros gradients.
pub fn (mut o NAdamOptimizer[T]) update() ! {
	o.step++
	mu_t := o.beta1 * (1.0 - 0.5 * math.pow(0.96, f64(o.step) * o.momentum_decay))
	mu_next := o.beta1 * (1.0 - 0.5 * math.pow(0.96, f64(o.step + 1) * o.momentum_decay))
	mu_product_t := o.mu_product_t * mu_t
	step := NAdamStepParams{
		beta1:           o.beta1
		beta2:           o.beta2
		beta1_t:         o.beta1_t * o.beta1
		beta2_t:         o.beta2_t * o.beta2
		mu_product_t:    mu_product_t
		mu_product_t1:   mu_product_t * mu_next
		mu_t:            mu_t
		mu_next:         mu_next
		momentum_decay:  o.momentum_decay
		step:            o.step
		lr:              o.learning_rate
		epsilon:         o.epsilon
		weight_decay:    o.weight_decay
		decoupled_decay: o.decoupled_weight_decay
	}
	o.mu_product_t = mu_product_t
	o.beta1_t *= o.beta1
	o.beta2_t *= o.beta2
	for i, mut parameter in o.params {
		if parameter.requires_grad {
			$if sizeof(T) == 8 {
				grad := parameter.grad.to_array()
				mut theta := parameter.value.to_array()
				mut m := o.first_moments[i].to_array()
				mut v := o.second_moments[i].to_array()
				nadam_step_f64_cpu(grad, mut theta, mut m, mut v, step)
				parameter.value = vtl.from_array(theta, parameter.value.shape) or { return err }
				o.first_moments[i] = vtl.from_array(m, parameter.value.shape) or { return err }
				o.second_moments[i] = vtl.from_array(v, parameter.value.shape) or { return err }
			} $else $if sizeof(T) == 4 {
				grad := parameter.grad.to_array()
				mut theta := parameter.value.to_array()
				mut m := o.first_moments[i].to_array()
				mut v := o.second_moments[i].to_array()
				nadam_step_f32_cpu(grad, mut theta, mut m, mut v, step)
				parameter.value = vtl.from_array(theta, parameter.value.shape) or { return err }
				o.first_moments[i] = vtl.from_array(m, parameter.value.shape) or { return err }
				o.second_moments[i] = vtl.from_array(v, parameter.value.shape) or { return err }
			} $else {
				return error('NAdamOptimizer.update: unsupported element type size ${sizeof(T)}')
			}
			parameter.grad = vtl.zeros_like[T](parameter.value)
		}
	}
}
