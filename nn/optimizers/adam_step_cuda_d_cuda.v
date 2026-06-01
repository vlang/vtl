module optimizers

import vsl.cuda
import vsl.cuda.compute
import vtl.autograd_cuda

// adam_step_f64 with DeviceSession persistent m/v/theta on GPU (#106).
pub fn adam_step_f64(grad []f64, mut theta []f64, mut m []f64, mut v []f64, p AdamStepParams,
	mut session &autograd_cuda.DeviceSession, slot int) {
	if !autograd_cuda.cuda_optimizer_enabled() || !session.enabled {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	dev := cuda.get_default_device() or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	n := grad.len
	mut sl := session.ensure_opt_slot(slot, n) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	if !sl.gpu_sync {
		sl.d_m.upload(m) or {
			adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
			return
		}
		sl.d_v.upload(v) or {
			adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
			return
		}
		sl.d_theta.upload(theta) or {
			adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
			return
		}
		sl.gpu_sync = true
	}
	sl.d_g.upload(grad) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	// m = beta1*m + (1-beta1)*g
	compute.gpu_buf_f64_dscal(dev, mut sl.d_m, n, p.beta1) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	compute.gpu_buf_f64_axpy(dev, 1.0 - p.beta1, &sl.d_g, mut sl.d_m, n) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	// v = beta2*v + (1-beta2)*g^2
	compute.gpu_buf_f64_mul_vec(dev, &sl.d_g, &sl.d_g, mut sl.d_work, n) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	compute.gpu_buf_f64_dscal(dev, mut sl.d_v, n, p.beta2) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	compute.gpu_buf_f64_axpy(dev, 1.0 - p.beta2, &sl.d_work, mut sl.d_v, n) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	// theta -= lr * m / (sqrt(v) + eps) — d_work = sqrt(v)+eps, then m/d_work, axpy to theta
	compute.gpu_buf_f64_copy(mut sl.d_work, &sl.d_v, n) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	compute.gpu_buf_f64_sqrt_inplace(mut sl.d_work, n) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	compute.gpu_buf_f64_add_scalar_inplace(mut sl.d_work, n, p.epsilon) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	// inv_denom on host (small sync), upload to d_work, m/denom -> d_work, theta -= lr * d_work
	mut inv_host := []f64{len: n}
	sl.d_work.download(mut inv_host) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	for i in 0 .. n {
		inv_host[i] = 1.0 / inv_host[i]
	}
	sl.d_work.upload(inv_host) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	compute.gpu_buf_f64_mul_vec(dev, &sl.d_m, &sl.d_work, mut sl.d_work, n) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	neg_lr := -p.lr_t
	compute.gpu_buf_f64_dscal(dev, mut sl.d_work, n, neg_lr) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	compute.gpu_buf_f64_axpy(dev, 1.0, &sl.d_work, mut sl.d_theta, n) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	// Sync back to CPU tensors (checkpointing / serialization)
	sl.d_m.download(mut m) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	sl.d_v.download(mut v) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	sl.d_theta.download(mut theta) or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
}
