module gate_iso

struct Tensor[T] {
mut:
	data []T
}

struct Payload[T] {
mut:
	t &Tensor[T]
}

interface Gate[T] {
	backward(p &Payload[T]) ![]&Tensor[T]
}

struct AddGate[T] {}

fn (g &AddGate[T]) backward(p &Payload[T]) ![]&Tensor[T] {
	return [&Tensor[T]{data: p.t.data}]
}

fn test_gate_f32() {
	mut g := &AddGate[f32]{}
	mut p := &Payload[f32]{t: &Tensor[f32]{data: [f32(1.0)]}}
	_ := Gate[f32](g)
	_ = g.backward(p)!
}
