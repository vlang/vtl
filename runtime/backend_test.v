module runtime

import os
import vsl.compute
import vtl.autograd

fn test_backend_from_string() {
	assert backend_from_string('auto') or { compute.Backend.cpu } == .auto
	assert backend_from_string('cpu') or { compute.Backend.auto } == .cpu
	assert backend_from_string('vulkan') or { compute.Backend.auto } == .vulkan
	assert backend_from_string('vcl') or { compute.Backend.auto } == .vcl
	if _ := backend_from_string('invalid') {
		assert false
	}
}

fn test_apply_policy_updates_context() {
	mut ctx := autograd.ctx[f64]()
	apply_policy[f64](mut ctx, ExecutionPolicy{
		backend: .vcl
		strict:  true
	})
	assert ctx.compute_backend == .vcl
	assert ctx.compute_strict
}

fn test_policy_from_env() {
	os.setenv('VTL_BACKEND', 'cpu', true)
	os.setenv('VTL_BACKEND_STRICT', '1', true)
	defer {
		os.unsetenv('VTL_BACKEND')
		os.unsetenv('VTL_BACKEND_STRICT')
	}
	policy := policy_from_env() or { panic(err) }
	assert policy.backend == .cpu
	assert policy.strict
}
