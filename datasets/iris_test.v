// vtest flaky
// vtest retry: 3
module datasets

fn test_iris() {
	iris := load_iris()!

	assert iris.features.shape == [150, 4]
	assert iris.labels.shape == [150]
}
