module optimizers

import vtl
import vtl.autograd
import vtl.nn.models
import rand
import math

fn ctx[T]() &autograd.Context[T] {
	return autograd.ctx[T]()
}

fn make_var[T](c &autograd.Context[T], val f64) &autograd.Variable[T] {
	t := vtl.from_1d([vtl.cast[T](val)]) or { panic(err) }
	return c.variable(t)
}

// SGD: param -= lr * grad
fn test_sgd_update() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 5.0)
	mut opt := sgd[f32](learning_rate: 0.1)
	opt.params << p
	p.grad.set_nth(0, 1.0)
	opt.update()!
	new_val := p.value.get_nth(0)
	// 5.0 - 0.1 * 1.0 = 4.9
	assert new_val - 4.9 < 1e-9 && new_val - 4.9 > -1e-9, 'SGD update expected 4.9, got ${new_val}'
}

fn test_sgd_zeros_grad_after_update() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 2.0)
	mut opt := sgd[f32](learning_rate: 0.5)
	opt.params << p
	p.grad.set_nth(0, 4.0)
	opt.update()!
	g := p.grad.get_nth(0)
	assert g == 0.0, 'grad should be zeroed after update, got ${g}'
}

fn test_adam_step_f32_cpu_moves_theta() {
	grad := [f32(1.0), f32(2.0)]
	mut theta := [f32(5.0), f32(5.0)]
	mut m := [f32(0.0), f32(0.0)]
	mut v := [f32(0.0), f32(0.0)]
	adam_step_f32_cpu(grad, mut theta, mut m, mut v, AdamStepParams{
		beta1:   0.9
		beta2:   0.999
		lr_t:    0.001
		epsilon: 1e-8
	})
	assert theta[0] < 5.0
	assert m[0] > 0.0
}

fn test_adam_step_f64_cpu() {
	grad := [1.0, 2.0]
	mut theta := [5.0, 5.0]
	mut m := [0.0, 0.0]
	mut v := [0.0, 0.0]
	adam_step_f64_cpu(grad, mut theta, mut m, mut v, AdamStepParams{
		beta1:   0.9
		beta2:   0.999
		lr_t:    0.001
		epsilon: 1e-8
	})
	assert theta[0] < 5.0
	assert m[0] > 0.0
}

fn test_adam_update_moves_param() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 1.0)
	mut opt := adam_optimizer[f32](learning_rate: 0.01)
	opt.params << p
	opt.first_moments << vtl.zeros_like[f32](p.grad)
	opt.second_moments << vtl.zeros_like[f32](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'Adam should decrease param when grad > 0'
}

fn test_adamw_update_moves_param() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 1.0)
	mut opt := adamw[f32](learning_rate: 0.01)
	opt.params << p
	opt.first_moments << vtl.zeros_like[f32](p.grad)
	opt.second_moments << vtl.zeros_like[f32](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'AdamW should decrease param when grad > 0, before=${before} after=${after}'
}

fn test_rmsprop_update_moves_param() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 1.0)
	mut opt := rmsprop[f32](learning_rate: 0.01)
	opt.params << p
	opt.sq_avg << vtl.zeros_like[f32](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'RMSProp should decrease param when grad > 0, before=${before} after=${after}'
}

fn test_adagrad_update_moves_param() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 1.0)
	mut opt := adagrad[f32](learning_rate: 0.1)
	opt.params << p
	opt.accumulated_sq_grads << vtl.zeros_like[f32](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'Adagrad should decrease param when grad > 0, before=${before} after=${after}'
}

fn test_nadam_step_f64_matches_reference() {
	beta1 := 0.9
	beta2 := 0.999
	decay := 0.004
	mu_t := beta1 * (1.0 - 0.5 * math.pow(0.96, decay))
	mu_next := beta1 * (1.0 - 0.5 * math.pow(0.96, 2.0 * decay))
	product_t := mu_t
	product_t1 := product_t * mu_next
	m_hat := mu_next * 0.1 / (1.0 - product_t1) + (1.0 - mu_t) / (1.0 - product_t)
	grad := [1.0]
	mut theta := [5.0]
	mut m := [0.0]
	mut v := [0.0]
	nadam_step_f64_cpu(grad, mut theta, mut m, mut v, NAdamStepParams{
		beta1:          beta1
		beta2:          beta2
		beta1_t:        beta1
		beta2_t:        beta2
		mu_product_t:   product_t
		mu_product_t1:  product_t1
		mu_t:           mu_t
		mu_next:        mu_next
		momentum_decay: decay
		step:           1
		lr:             0.001
		epsilon:        1e-8
	})
	assert math.abs(m[0] - 0.1) < 1e-12
	assert math.abs(v[0] - 0.001) < 1e-12
	assert math.abs(theta[0] - (5.0 - 0.001 * m_hat / (1.0 + 1e-8))) < 1e-12
}

fn test_nadam_step_f32_matches_reference() {
	beta1 := 0.9
	beta2 := 0.999
	decay := 0.004
	mu_t := beta1 * (1.0 - 0.5 * math.pow(0.96, decay))
	mu_next := beta1 * (1.0 - 0.5 * math.pow(0.96, 2.0 * decay))
	product_t := mu_t
	product_t1 := product_t * mu_next
	m_hat := mu_next * 0.1 / (1.0 - product_t1) + (1.0 - mu_t) / (1.0 - product_t)
	grad := [f32(1.0)]
	mut theta := [f32(5.0)]
	mut m := [f32(0.0)]
	mut v := [f32(0.0)]
	nadam_step_f32_cpu(grad, mut theta, mut m, mut v, NAdamStepParams{
		beta1:          beta1
		beta2:          beta2
		beta1_t:        beta1
		beta2_t:        beta2
		mu_product_t:   product_t
		mu_product_t1:  product_t1
		mu_t:           mu_t
		mu_next:        mu_next
		momentum_decay: decay
		step:           1
		lr:             0.001
		epsilon:        1e-8
	})
	assert math.abs(f64(m[0]) - 0.1) < 1e-6
	assert math.abs(f64(v[0]) - 0.001) < 1e-6
	assert math.abs(f64(theta[0]) - (5.0 - 0.001 * m_hat / (1.0 + 1e-8))) < 1e-6
}

fn test_radam_step_f64_unrectified_reference() {
	grad := [1.0]
	mut theta := [5.0]
	mut m := [0.0]
	mut v := [0.0]
	radam_step_f64_cpu(grad, mut theta, mut m, mut v, RAdamStepParams{
		beta1:   0.9
		beta2:   0.999
		beta1_t: 0.9
		beta2_t: 0.999
		step:    1
		lr:      0.001
		epsilon: 1e-8
	})
	assert math.abs(m[0] - 0.1) < 1e-12
	assert math.abs(v[0] - 0.001) < 1e-12
	assert math.abs(theta[0] - 4.999) < 1e-12
}

fn test_radam_step_f32_unrectified_reference() {
	grad := [f32(1.0)]
	mut theta := [f32(5.0)]
	mut m := [f32(0.0)]
	mut v := [f32(0.0)]
	radam_step_f32_cpu(grad, mut theta, mut m, mut v, RAdamStepParams{
		beta1:   0.9
		beta2:   0.999
		beta1_t: 0.9
		beta2_t: 0.999
		step:    1
		lr:      0.001
		epsilon: 1e-8
	})
	assert math.abs(f64(m[0]) - 0.1) < 1e-6
	assert math.abs(f64(v[0]) - 0.001) < 1e-6
	assert math.abs(f64(theta[0]) - 4.999) < 1e-6
}

fn test_radam_rectified_step_reference() {
	beta1 := 0.9
	beta2 := 0.999
	step := 6
	grad := [0.0]
	mut theta := [5.0]
	mut m := [1.0 - math.pow(beta1, 5.0)]
	mut v := [1.0 - math.pow(beta2, 5.0)]
	beta1_t := math.pow(beta1, 6.0)
	beta2_t := math.pow(beta2, 6.0)
	rho_inf := 2.0 / (1.0 - beta2) - 1.0
	rho_t := rho_inf - 2.0 * f64(step) * beta2_t / (1.0 - beta2_t)
	rectification := math.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
	m_hat := (beta1 * m[0]) / (1.0 - beta1_t)
	v_t := beta2 * v[0]
	want := 5.0 - 0.001 * rectification * math.sqrt(1.0 - beta2_t) * m_hat / (math.sqrt(v_t) + 1e-8)
	radam_step_f64_cpu(grad, mut theta, mut m, mut v, RAdamStepParams{
		beta1:   beta1
		beta2:   beta2
		beta1_t: beta1_t
		beta2_t: beta2_t
		step:    step
		lr:      0.001
		epsilon: 1e-8
	})
	assert rho_t > 5.0
	assert math.abs(theta[0] - want) < 1e-12
}

fn test_nadam_and_radam_optimizer_updates_zero_gradients() ! {
	for use_nadam in [true, false] {
		c := ctx[f64]()
		mut p := make_var[f64](c, 1.0)
		if use_nadam {
			mut opt := nadam_optimizer[f64](learning_rate: 0.01)
			opt.params << p
			opt.first_moments << vtl.zeros_like[f64](p.grad)
			opt.second_moments << vtl.zeros_like[f64](p.grad)
			for _ in 0 .. 20 {
				p.grad.set_nth(0, 1.0)
				opt.update()!
			}
		} else {
			mut opt := radam_optimizer[f64](learning_rate: 0.01)
			opt.params << p
			opt.first_moments << vtl.zeros_like[f64](p.grad)
			opt.second_moments << vtl.zeros_like[f64](p.grad)
			for _ in 0 .. 20 {
				p.grad.set_nth(0, 1.0)
				opt.update()!
			}
		}
		assert p.value.get_nth(0) < 1.0
		assert p.grad.get_nth(0) == 0.0
	}
}

fn test_nadam_and_radam_f32_optimizer_updates() ! {
	for use_nadam in [true, false] {
		c := ctx[f32]()
		mut p := make_var[f32](c, 1.0)
		if use_nadam {
			mut opt := nadam_optimizer[f32](learning_rate: 0.01)
			opt.params << p
			opt.first_moments << vtl.zeros_like[f32](p.grad)
			opt.second_moments << vtl.zeros_like[f32](p.grad)
			p.grad.set_nth(0, 1.0)
			opt.update()!
			assert opt.step == 1
			assert opt.beta1_t == 0.9
			assert opt.mu_product_t > 0.0 && opt.mu_product_t < 1.0
		} else {
			mut opt := radam_optimizer[f32](learning_rate: 0.01)
			opt.params << p
			opt.first_moments << vtl.zeros_like[f32](p.grad)
			opt.second_moments << vtl.zeros_like[f32](p.grad)
			p.grad.set_nth(0, 1.0)
			opt.update()!
			assert opt.step == 1
			assert math.abs(opt.beta1_t - 0.9) < 1e-12
		}
		assert p.value.get_nth(0) < 1.0
		assert p.grad.get_nth(0) == 0.0
	}
}

fn test_nadam_and_radam_weight_decay_modes() ! {
	c1 := ctx[f64]()
	mut nadam_coupled_param := make_var[f64](c1, 2.0)
	mut nadam_coupled := nadam_optimizer[f64](learning_rate: 0.01, weight_decay: 0.1)
	nadam_coupled.params << nadam_coupled_param
	nadam_coupled.first_moments << vtl.zeros_like[f64](nadam_coupled_param.grad)
	nadam_coupled.second_moments << vtl.zeros_like[f64](nadam_coupled_param.grad)
	nadam_coupled.update()!
	assert nadam_coupled_param.value.get_nth(0) < 2.0
	assert nadam_coupled.first_moments[0].get_nth(0) > 0.0
	assert nadam_coupled.second_moments[0].get_nth(0) > 0.0

	c2 := ctx[f64]()
	mut nadam_decoupled_param := make_var[f64](c2, 2.0)
	mut nadam_decoupled := nadam_optimizer[f64](
		learning_rate:          0.01
		weight_decay:           0.1
		decoupled_weight_decay: true
	)
	nadam_decoupled.params << nadam_decoupled_param
	nadam_decoupled.first_moments << vtl.zeros_like[f64](nadam_decoupled_param.grad)
	nadam_decoupled.second_moments << vtl.zeros_like[f64](nadam_decoupled_param.grad)
	nadam_decoupled.update()!
	assert math.abs(nadam_decoupled_param.value.get_nth(0) - 1.998) < 1e-12
	assert nadam_decoupled.first_moments[0].get_nth(0) == 0.0
	assert nadam_decoupled.second_moments[0].get_nth(0) == 0.0

	c3 := ctx[f64]()
	mut radam_coupled_param := make_var[f64](c3, 2.0)
	mut radam_coupled := radam_optimizer[f64](learning_rate: 0.01, weight_decay: 0.1)
	radam_coupled.params << radam_coupled_param
	radam_coupled.first_moments << vtl.zeros_like[f64](radam_coupled_param.grad)
	radam_coupled.second_moments << vtl.zeros_like[f64](radam_coupled_param.grad)
	radam_coupled.update()!
	assert radam_coupled_param.value.get_nth(0) < 2.0
	assert radam_coupled.first_moments[0].get_nth(0) > 0.0
	assert radam_coupled.second_moments[0].get_nth(0) > 0.0

	c4 := ctx[f64]()
	mut radam_decoupled_param := make_var[f64](c4, 2.0)
	mut radam_decoupled := radam_optimizer[f64](
		learning_rate:          0.01
		weight_decay:           0.1
		decoupled_weight_decay: true
	)
	radam_decoupled.params << radam_decoupled_param
	radam_decoupled.first_moments << vtl.zeros_like[f64](radam_decoupled_param.grad)
	radam_decoupled.second_moments << vtl.zeros_like[f64](radam_decoupled_param.grad)
	radam_decoupled.update()!
	assert math.abs(radam_decoupled_param.value.get_nth(0) - 1.998) < 1e-12
	assert radam_decoupled.first_moments[0].get_nth(0) == 0.0
	assert radam_decoupled.second_moments[0].get_nth(0) == 0.0
}

fn train_xor_with_nadam() !(f64, f64, int) {
	rand.seed([u32(42), u32(0)])
	c := ctx[f64]()
	mut model := models.sequential_from_ctx[f64](c)
	model.input([2])
	model.linear(4)
	model.tanh()
	model.linear(1)
	model.sigmoid_cross_entropy_loss()
	mut optimizer := nadam_optimizer[f64](learning_rate: 0.01)
	optimizer.build_params(model.info.layers)
	mut x := c.variable(vtl.from_array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])!)
	y := vtl.from_array([0.0, 1.0, 1.0, 0.0], [4])!
	mut initial_loss := 0.0
	for epoch in 0 .. 5000 {
		prediction := model.forward(x)!
		mut loss := model.loss(prediction, y)!
		if epoch == 0 { initial_loss = loss.value.get_nth(0) }
		loss.backprop()!
		optimizer.update()!
		x.grad = vtl.zeros_like[f64](x.value)
	}
	prediction := model.forward(x)!
	mut correct := 0
	for i in 0 .. 4 {
		if (prediction.value.get_nth(i) > 0.0) == (y.get_nth(i) == 1.0) { correct++ }
	}
	final_loss := model.loss(prediction, y)!.value.get_nth(0)
	return initial_loss, final_loss, correct
}

fn train_xor_with_radam() !(f64, f64, int) {
	rand.seed([u32(42), u32(0)])
	c := ctx[f64]()
	mut model := models.sequential_from_ctx[f64](c)
	model.input([2])
	model.linear(4)
	model.tanh()
	model.linear(1)
	model.sigmoid_cross_entropy_loss()
	mut optimizer := radam_optimizer[f64](learning_rate: 0.01)
	optimizer.build_params(model.info.layers)
	mut x := c.variable(vtl.from_array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])!)
	y := vtl.from_array([0.0, 1.0, 1.0, 0.0], [4])!
	mut initial_loss := 0.0
	for epoch in 0 .. 5000 {
		prediction := model.forward(x)!
		mut loss := model.loss(prediction, y)!
		if epoch == 0 { initial_loss = loss.value.get_nth(0) }
		loss.backprop()!
		optimizer.update()!
		x.grad = vtl.zeros_like[f64](x.value)
	}
	prediction := model.forward(x)!
	mut correct := 0
	for i in 0 .. 4 {
		if (prediction.value.get_nth(i) > 0.0) == (y.get_nth(i) == 1.0) { correct++ }
	}
	final_loss := model.loss(prediction, y)!.value.get_nth(0)
	return initial_loss, final_loss, correct
}

fn test_nadam_and_radam_converge_on_deterministic_xor() ! {
	nadam_initial, nadam_final, nadam_correct := train_xor_with_nadam()!
	assert nadam_correct == 4, 'NAdam should classify all XOR cases, got ${nadam_correct}/4'
	assert nadam_final < nadam_initial, 'NAdam loss did not decrease: ${nadam_initial} -> ${nadam_final}'
	radam_initial, radam_final, radam_correct := train_xor_with_radam()!
	assert radam_correct == 4, 'RAdam should classify all XOR cases, got ${radam_correct}/4'
	assert radam_final < radam_initial, 'RAdam loss did not decrease: ${radam_initial} -> ${radam_final}'
}
