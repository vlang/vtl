import vtl

fn main() {
	flags := vtl.from_1d([0b0001, 0b0101, 0b1100, 0b1111])!
	feature_mask := vtl.from_1d([0b0100])!
	has_feature := flags.bitwise_and(feature_mask)!
	assert has_feature.to_array() == [0, 4, 4, 4]
	println('Masked flags: ${has_feature.to_array()}')
	println('Shifted flags: ${flags.left_shift(1)!.to_array()}')
}
