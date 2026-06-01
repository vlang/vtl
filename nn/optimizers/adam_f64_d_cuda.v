module optimizers

import vtl
import vtl.autograd
import vtl.autograd_cuda

// adam_update_f64_cuda exposes this operation as part of the public API.
pub fn adam_update_f64_cuda(param voidptr, m_tensor voidptr, v_tensor voidptr, step AdamStepParams, slot int) ! {
	mut v := unsafe { &autograd.Variable[f64](param) }
	mut m := unsafe { &vtl.Tensor[f64](m_tensor) }
	mut v_mom := unsafe { &vtl.Tensor[f64](v_tensor) }
	grad := v.grad.to_array()
	mut theta := v.value.to_array()
	mut m_arr := m.to_array()
	mut v_arr := v_mom.to_array()
	adam_step_f64(grad, mut theta, mut m_arr, mut v_arr, step, v.context.device_session, slot)
	v.value = vtl.from_array(theta, v.value.shape)!
	m = vtl.from_array(m_arr, v.value.shape)!
	v_mom = vtl.from_array(v_arr, v.value.shape)!
}
