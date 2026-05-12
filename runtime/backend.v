module runtime

import os
import vsl.compute
import vtl.autograd

// ExecutionPolicy configures runtime backend selection.
pub struct ExecutionPolicy {
pub:
	backend compute.Backend = .auto
	strict  bool
}

// available_backends returns backends compiled in the current binary.
pub fn available_backends() []compute.Backend {
	return compute.available_backends()
}

// apply_policy applies runtime backend settings to an autograd context.
pub fn apply_policy[T](mut ctx autograd.Context[T], policy ExecutionPolicy) {
	ctx.set_compute_backend(policy.backend)
	ctx.set_compute_strict(policy.strict)
}

// backend_from_string parses backend names used by env/config files.
pub fn backend_from_string(value string) !compute.Backend {
	match value.to_lower() {
		'', 'auto' { return .auto }
		'cpu' { return .cpu }
		'vulkan', 'vk' { return .vulkan }
		'vcl', 'opencl', 'ocl' { return .vcl }
		'cuda' { return error('cuda backend is not available in this VTL build yet') }
		else { return error('unknown backend `${value}`') }
	}
}

// policy_from_env reads backend options from env vars.
// - VTL_BACKEND: auto|cpu|vulkan|vcl|cuda
// - VTL_BACKEND_STRICT: 1|true|yes to disable fallback
pub fn policy_from_env() !ExecutionPolicy {
	backend := backend_from_string(os.getenv_opt('VTL_BACKEND') or { '' })!
	strict_value := os.getenv_opt('VTL_BACKEND_STRICT') or { '' }
	strict := strict_value.to_lower() in ['1', 'true', 'yes', 'on']
	return ExecutionPolicy{
		backend: backend
		strict:  strict
	}
}
