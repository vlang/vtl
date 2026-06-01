module autograd

import vsl.la as vsl_la
import vtl

fn gate_matmul[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[T] {
	a.assert_matrix()!
	b.assert_matrix()!
	if a.shape[1] != b.shape[0] {
		return error('Invalid shapes for matrix multiplication ${a.shape} and ${b.shape}')
	}
	ma := a.copy(.row_major)
	mb := b.copy(.row_major)
	mut dm := vsl_la.Matrix.new[f64](a.shape[0], b.shape[1])
	mam := vsl_la.Matrix.raw(a.shape[0], a.shape[1], ma.as_f64().to_array())
	mbm := vsl_la.Matrix.raw(b.shape[0], b.shape[1], mb.as_f64().to_array())
	vsl_la.matrix_matrix_mul(mut dm, 1.0, mam, mbm)
	res := vtl.from_2d[f64](dm.get_deep2())!
	if sizeof(T) == 4 {
		return unsafe { &vtl.Tensor[T](res.as_f32()) }
	}
	return unsafe { &vtl.Tensor[T](res) }
}

// MatMulGate defines a public data structure for this module.
pub struct MatMulGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
	b &Variable[T] = unsafe { nil }
}

// matmul_gate exposes this operation as part of the public API.
pub fn matmul_gate[T](a &Variable[T], b &Variable[T]) &MatMulGate[T] {
	return &MatMulGate[T]{
		a: a
		b: b
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &MatMulGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gate_matmul[T](gradient, g.b.value.t()!)!
	r1 := gate_matmul[T](g.a.value.t()!, gradient)!
	return [r0, r1]
}

// cache exposes this operation as part of the public API.
pub fn (g &MatMulGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]

	match a {
		Variable[T] {
			match b {
				Variable[T] {
					result.grad = vtl.zeros_like[T](result.value)
					result.requires_grad = true

					register[T]('MatMul', g, result, [a, b])!
				}
				else {
					return error('MatMulGate: b must be a Variable')
				}
			}
		}
		else {
			return error('MatMulGate: a must be a Variable')
		}
	}
}
