module amp_test

import math
import vtl
import vtl.amp

fn test_scale_and_unscale_round_trip() ! {
	mut scaler := amp.new(amp.GradScalerConfig{ init_scale: 4.0, growth_interval: 2 })!
	assert scaler.scale_loss(3.0) == 12.0
	assert scaler.scale_factor() == 4.0

	mut grad := vtl.from_array([f64(8), -4, 0], [3])!
	mut gradients := [grad]
	assert scaler.unscale(mut gradients)!
	assert grad.to_array() == [f64(2), -1, 0]
}

fn test_unscale_detects_overflow_and_leaves_values_untouched() ! {
	mut scaler := amp.new(amp.GradScalerConfig{ init_scale: 4.0 })!
	mut fine := vtl.from_array([f64(8)], [1])!
	mut overflowing := vtl.from_array([f64(2), math.inf(1)], [2])!
	mut gradients := [fine, overflowing]
	assert !scaler.unscale(mut gradients)!
	assert fine.to_array() == [f64(2)]
	assert math.is_finite(overflowing.get_nth[f64](0))
	assert math.is_inf(overflowing.get_nth[f64](1), 1)
}

fn test_update_grows_after_consecutive_good_steps() ! {
	mut scaler := amp.new(amp.GradScalerConfig{
		init_scale:      4.0
		growth_factor:   2.0
		growth_interval: 2
	})!
	scaler.update(false)
	assert scaler.scale_factor() == 4.0
	assert scaler.state().good_steps == 1
	scaler.update(false)
	assert scaler.scale_factor() == 8.0
	assert scaler.state().good_steps == 0
}

fn test_update_backs_off_on_overflow() ! {
	mut scaler := amp.new(amp.GradScalerConfig{
		init_scale:      8.0
		backoff_factor:  0.5
		growth_interval: 1
	})!
	scaler.update(false)
	assert scaler.scale_factor() == 16.0
	scaler.update(true)
	assert scaler.scale_factor() == 8.0
	assert scaler.state().good_steps == 0
}

fn test_disabled_scaler_is_a_no_op() ! {
	mut scaler := amp.new(amp.GradScalerConfig{ enabled: false })!
	assert scaler.scale_loss(3.0) == 3.0
	mut grad := vtl.from_array([f64(8)], [1])!
	mut gradients := [grad]
	assert scaler.unscale(mut gradients)!
	assert grad.get_nth[f64](0) == 8
	scaler.update(true)
	assert scaler.scale_factor() == 1.0
}

fn test_invalid_config_is_rejected() ! {
	amp.new(amp.GradScalerConfig{ init_scale: 0.0 }) or {
		assert err.msg().contains('init_scale')
		return
	}
	assert false, 'zero init_scale should be rejected'
}

fn test_state_round_trip() ! {
	mut scaler := amp.new(amp.GradScalerConfig{ init_scale: 4.0, growth_interval: 3 })!
	scaler.update(false)
	state := scaler.state()
	restored := amp.from_state(state, amp.GradScalerConfig{ init_scale: 4.0, growth_interval: 3 })!
	assert restored.state().scale == state.scale
	assert restored.state().good_steps == state.good_steps
	assert restored.state().enabled == state.enabled
}

fn test_from_state_rejects_invalid_state() ! {
	amp.from_state(amp.GradScalerState{ scale: 0, good_steps: 0, enabled: true }, amp.GradScalerConfig{}) or {
		assert err.msg().contains('restored scale')
		return
	}
	assert false, 'zero restored scale should be rejected'
}

fn test_unscale_detects_post_op_overflow_and_leaves_value_untouched() ! {
	// Finite f32 input whose unscaled result overflows f32: a naive
	// implementation would store +inf, the fixed one leaves it unchanged.
	mut scaler := amp.new(amp.GradScalerConfig{ init_scale: 0.5 })!
	large := f32(3.0e38)
	assert math.is_finite(f64(large))
	mut grad := vtl.from_array([large], [1])!
	mut gradients := [grad]
	assert !scaler.unscale(mut gradients)!
	stored := grad.get_nth[f32](0)
	assert math.is_finite(f64(stored)), 'post-op overflow must not store inf'
	assert stored == large
}

fn test_disabled_state_round_trip_restores_disabled() ! {
	mut scaler := amp.new(amp.GradScalerConfig{ enabled: false })!
	state := scaler.state()
	assert state.enabled == false
	// Saved disabled flag wins even when the restore config is enabled.
	restored := amp.from_state(state, amp.GradScalerConfig{ init_scale: 4.0, growth_interval: 3 })!
	assert restored.state().enabled == false
	assert restored.scale_loss(3.0) == 3.0
	assert restored.state().scale == state.scale
}
