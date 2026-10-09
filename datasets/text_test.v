module datasets

fn test_tokenize_text_normalizes_case_and_punctuation() {
	assert tokenize_text('A joyful, MOVING review!') == ['a', 'joyful', 'moving', 'review']
	assert tokenize_text('  An\texcellent\nstory. ') == ['an', 'excellent', 'story']
	assert tokenize_text('Café déjà vu') == ['café', 'déjà', 'vu']
	unicode_whitespace := [
		'\u0085',
		'\u00a0',
		'\u1680',
		'\u2000',
		'\u2001',
		'\u2002',
		'\u2003',
		'\u2004',
		'\u2005',
		'\u2006',
		'\u2007',
		'\u2008',
		'\u2009',
		'\u200a',
		'\u2028',
		'\u2029',
		'\u202f',
		'\u205f',
		'\u3000',
	]
	for separator in unicode_whitespace {
		assert tokenize_text('good${separator}movie') == ['good', 'movie']
	}
}

fn test_text_vocabulary_frequency_order_and_stable_ties() ! {
	vocabulary := build_text_vocabulary(['Blue red blue', 'green red'], 0, 1)!
	assert vocabulary.tokens == ['<pad>', '<unk>', 'blue', 'red', 'green']
	assert vocabulary.index('blue') == 2
	assert vocabulary.index('missing') == 1
}

fn test_text_vocabulary_limits_and_unknown_round_trip() ! {
	vocabulary := build_text_vocabulary(['pear apple pear plum'], 4, 1)!
	assert vocabulary.tokens == ['<pad>', '<unk>', 'pear', 'apple']
	encoded := vocabulary.encode('PEAR plum unknown')
	assert encoded == [2, 1, 1]
	assert vocabulary.decode(encoded) == ['pear', '<unk>', '<unk>']
	assert vocabulary.decode([-1, 100]) == ['<unk>', '<unk>']
}

fn test_text_vocabulary_frequency_threshold_and_invalid_options() ! {
	vocabulary := build_text_vocabulary(['one two two'], 0, 2)!
	assert vocabulary.tokens == ['<pad>', '<unk>', 'two']
	_ := build_text_vocabulary(['a'], 1, 1) or {
		assert err.msg().contains('max_size')
		return
	}
	assert false, 'expected an error for max_size below two'
	_ = build_text_vocabulary(['a'], 0, 0) or {
		assert err.msg().contains('min_frequency')
		return
	}
	assert false, 'expected an error for min_frequency below one'
}
