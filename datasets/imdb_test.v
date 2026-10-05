module datasets

import os

fn write_imdb_fixture_review(root string, split string, sentiment string, file_name string, text string) ! {
	dir := os.join_path(root, split, sentiment)
	os.mkdir_all(dir)!
	os.write_file(os.join_path(dir, file_name), text)!
}

fn create_imdb_fixture(root string) ! {
	write_imdb_fixture_review(root, 'train', 'pos', '0_10.txt', 'A joyful and moving review.')!
	write_imdb_fixture_review(root, 'train', 'neg', '0_1.txt', 'A dull and disappointing review.')!
	write_imdb_fixture_review(root, 'test', 'pos', '0_9.txt', 'An entertaining story.')!
	write_imdb_fixture_review(root, 'test', 'pos', '1_10.txt', 'An excellent performance.')!
	write_imdb_fixture_review(root, 'test', 'neg', '0_1.txt', 'A tedious story.')!
	write_imdb_fixture_review(root, 'test', 'neg', '1_2.txt', 'A weak performance.')!
}

fn test_imdb_fixture_subset_labels_and_review_text() ! {
	root := os.join_path(os.temp_dir(), 'vtl-imdb-fixture-${os.getpid()}')
	os.mkdir_all(root)!
	defer {
		os.rmdir_all(root) or {}
	}
	create_imdb_fixture(root)!

	dataset := load_imdb_from_dir(root, ImdbConfig{
		train_count: 2
		test_count:  4
	})!

	assert dataset.train_features.shape == [2]
	assert dataset.train_labels.shape == [2]
	assert dataset.test_features.shape == [4]
	assert dataset.test_labels.shape == [4]
	assert dataset.train_features.get_nth(0) == 'A joyful and moving review.'
	assert dataset.train_features.get_nth(1) == 'A dull and disappointing review.'
	assert dataset.train_labels.get_nth(0) == 1
	assert dataset.train_labels.get_nth(1) == 0
	assert dataset.test_labels.get_nth(0) == 1
	assert dataset.test_labels.get_nth(1) == 1
	assert dataset.test_labels.get_nth(2) == 0
	assert dataset.test_labels.get_nth(3) == 0
}

fn test_imdb_subset_is_balanced_and_deterministic() ! {
	root := os.join_path(os.temp_dir(), 'vtl-imdb-subset-${os.getpid()}')
	os.mkdir_all(root)!
	defer {
		os.rmdir_all(root) or {}
	}
	create_imdb_fixture(root)!

	dataset := load_imdb_from_dir(root, ImdbConfig{
		train_count: 1
		test_count:  2
	})!
	assert dataset.train_labels.shape == [1]
	assert dataset.train_labels.get_nth(0) == 1
	assert dataset.test_labels.shape == [2]
	assert dataset.test_labels.get_nth(0) == 1
	assert dataset.test_labels.get_nth(1) == 0
}

fn test_imdb_missing_cache_returns_error_without_network() {
	root := os.join_path(os.temp_dir(), 'vtl-imdb-missing-${os.getpid()}')
	_ := load_imdb_from_dir(root, ImdbConfig{
		train_count: 2
		test_count:  2
	}) or {
		assert err.msg().contains('invalid cached IMDB train split')
		return
	}
	assert false, 'expected missing cached reviews to return an error'
}

fn test_imdb_default_config_uses_full_balanced_splits() {
	config := ImdbConfig{}
	assert config.train_count == 25000
	assert config.test_count == 25000
}
