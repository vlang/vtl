module etype

fn test_default() {
	assert default_type == typeof(default_init)
}

fn test_f64_init() {
	v := Num(1.0)
	if v is f64 {
		assert v == 1.0
	} else {
		panic('This should never happen')
	}
}

fn test_as_type() {
	v := Num(1.0)
	f := v.as_type<f64>()
	assert v as f64 == f
}
