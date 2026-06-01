module autograd_cuda

import vsl.cuda
import vsl.cuda.compute

// DeviceOptimizerSlot holds persistent GPU buffers for one parameter tensor (Adam #106).
struct DeviceOptimizerSlot {
mut:
	d_m      compute.GpuBufF64
	d_v      compute.GpuBufF64
	d_g      compute.GpuBufF64
	d_work   compute.GpuBufF64
	d_theta  compute.GpuBufF64
	n        int
	gpu_sync bool // m,v,theta resident on GPU; skip H→D on next step
}

struct DeviceOptimizerState {
mut:
	slots []DeviceOptimizerSlot
}

fn (mut s DeviceSession) opt_state_mut() &DeviceOptimizerState {
	if s.optimizer_state == unsafe { nil } {
		s.optimizer_state = voidptr(&DeviceOptimizerState{})
	}
	return unsafe { &DeviceOptimizerState(s.optimizer_state) }
}

pub fn (mut s DeviceSession) ensure_opt_slot(slot int, n int) !&DeviceOptimizerSlot {
	st := s.opt_state_mut()
	for st.slots.len <= slot {
		st.slots << DeviceOptimizerSlot{}
	}
	mut sl := &st.slots[slot]
	sl.n = n
	sl.d_m.ensure(n)!
	sl.d_v.ensure(n)!
	sl.d_g.ensure(n)!
	sl.d_work.ensure(n)!
	sl.d_theta.ensure(n)!
	return sl
}

pub fn (mut s DeviceSession) reset_optimizer_state() {
	if s.optimizer_state != unsafe { nil } {
		mut st := unsafe { &DeviceOptimizerState(s.optimizer_state) }
		for mut sl in st.slots {
			sl.d_m.release()
			sl.d_v.release()
			sl.d_g.release()
			sl.d_work.release()
			sl.d_theta.release()
		}
		s.optimizer_state = unsafe { nil }
	}
}
