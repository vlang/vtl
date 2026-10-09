module datasets

pub const text_pad_token = '<pad>'
pub const text_unknown_token = '<unk>'

// TextVocabulary maps normalized text tokens to stable integer IDs.
// The first IDs are reserved for padding and out-of-vocabulary tokens.
pub struct TextVocabulary {
pub:
	tokens []string
mut:
	token_to_index map[string]int
}

struct TokenFrequency {
	token string
	count int
}

// tokenize_text lowercases text and splits on whitespace and ASCII punctuation.
// Non-ASCII characters are preserved so UTF-8 words remain intact.
pub fn tokenize_text(text string) []string {
	lowered := text.to_lower()
	mut tokens := []string{}
	mut token_start := -1
	mut i := 0
	for i < lowered.len {
		ch := lowered[i]
		whitespace_len := unicode_text_whitespace_len(lowered, i)
		if whitespace_len > 0 {
			if token_start >= 0 {
				tokens << lowered[token_start..i]
				token_start = -1
			}
			i += whitespace_len
			continue
		}
		if ch <= 127 && !(ch.is_letter() || ch.is_digit()) {
			if token_start >= 0 {
				tokens << lowered[token_start..i]
				token_start = -1
			}
		} else if token_start < 0 {
			token_start = i
		}
		i++
	}
	if token_start >= 0 {
		tokens << lowered[token_start..]
	}
	return tokens
}

// unicode_text_whitespace_len returns the UTF-8 byte length of a Unicode
// whitespace character at index, or zero when the bytes are not whitespace.
fn unicode_text_whitespace_len(text string, index int) int {
	if index + 1 < text.len && text[index] == 0xc2 && text[index + 1] in [0x85, 0xa0] {
		return 2
	}
	if index + 2 >= text.len {
		return 0
	}
	if text[index] == 0xe1 && text[index + 1] == 0x9a && text[index + 2] == 0x80 {
		return 3
	}
	if text[index] == 0xe2 {
		if text[index + 1] == 0x80 && (text[index + 2] in 0x80 .. 0x8b
			|| text[index + 2] in [0xa8, 0xa9, 0xaf]) {
			return 3
		}
		if text[index + 1] == 0x81 && text[index + 2] == 0x9f {
			return 3
		}
	}
	if text[index] == 0xe3 && text[index + 1] == 0x80 && text[index + 2] == 0x80 {
		return 3
	}
	return 0
}

// build_text_vocabulary builds a frequency-ranked vocabulary from texts.
// max_size includes the reserved <pad> and <unk> entries; zero means unlimited.
// Tokens with frequency below min_frequency are omitted. Equal-frequency tokens
// are ordered lexicographically to make the result deterministic.
pub fn build_text_vocabulary(texts []string, max_size int, min_frequency int) !TextVocabulary {
	if max_size != 0 && max_size < 2 {
		return error('text vocabulary max_size must be zero or at least 2')
	}
	if min_frequency < 1 {
		return error('text vocabulary min_frequency must be at least 1')
	}
	mut frequencies := map[string]int{}
	for text in texts {
		for token in tokenize_text(text) {
			frequencies[token]++
		}
	}
	mut ranked := []TokenFrequency{cap: frequencies.len}
	for token, count in frequencies {
		if count >= min_frequency && token !in [text_pad_token, text_unknown_token] {
			ranked << TokenFrequency{token, count}
		}
	}
	ranked.sort_with_compare(fn (a &TokenFrequency, b &TokenFrequency) int {
		if a.count > b.count {
			return -1
		}
		if a.count < b.count {
			return 1
		}
		return a.token.compare(b.token)
	})
	if max_size > 0 && ranked.len > max_size - 2 {
		ranked = ranked[..max_size - 2]
	}
	mut tokens := [text_pad_token, text_unknown_token]
	for item in ranked {
		tokens << item.token
	}
	mut token_to_index := map[string]int{}
	for index, token in tokens {
		token_to_index[token] = index
	}
	return TextVocabulary{
		tokens:         tokens
		token_to_index: token_to_index
	}
}

// index returns the ID for token, or the unknown-token ID when it is absent.
pub fn (v &TextVocabulary) index(token string) int {
	return v.token_to_index[token] or { 1 }
}

// encode tokenizes text and returns its vocabulary IDs.
pub fn (v &TextVocabulary) encode(text string) []int {
	return tokenize_text(text).map(v.index(it))
}

// decode converts IDs back to tokens. Invalid IDs decode as <unk>.
pub fn (v &TextVocabulary) decode(indices []int) []string {
	mut decoded := []string{cap: indices.len}
	for index in indices {
		if index >= 0 && index < v.tokens.len {
			decoded << v.tokens[index]
		} else {
			decoded << text_unknown_token
		}
	}
	return decoded
}
