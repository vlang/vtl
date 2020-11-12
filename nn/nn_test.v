import vtl.nn
import vtl.num

fn xor_test() {
	expected := num.from_f64_2d([[0.0387387],
		[0.976217], [0.976216], [0.00880164]])
	features := num.from_int([0, 0, 0, 1, 1, 0, 1, 1], [4, 2])
	labels := num.from_int([0, 1, 1, 0], [4, 1])
	mut m := nn.new(0.7, 10000, 3, .sigmoid)
	m.learn(features, labels)
	assert m.predict(features).str() == expected.str()
}
