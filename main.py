import tensorflow as tf

def get_target_and_size(sequence, target_index, window_size):
    target_word = sequence[target_index]
    half_window_size = window_size // 2
    return (target_word, half_window_size)

def get_window_indices(sequence, target_index, half_window_size):
    left_incl = max(0, target_index - half_window_size)
    right_excl = min(len(sequence), target_index + half_window_size + 1)
    return left_incl, right_excl

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Convert a list of text strings into word sequences
    def tokenize_text_corpus(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences

    # Convert a list of text strings into word sequences
    def get_target_and_context(self, sequence, target_index, window_size):
        target_word = sequence[target_index]
        half_window_size = window_size // 2
        left_incl = max(0, target_index - half_window_size)
        right_excl = min(len(sequence), target_index + half_window_size + 1)
        return target_word, left_incl, right_excl

    # Create (target, context) pairs for a given window size
    def create_target_context_pairs(self, texts, window_size):
        pairs = []
        sequences = self.tokenize_text_corpus(texts)
        for sequence in sequences:
            for i in range(len(sequence)):
                target_word, left_incl, right_excl = self.get_target_and_context(
                    sequence, i, window_size)
                for j in range(left_incl, right_excl):
                    if j != i:
                        pairs.append((target_word, sequence[j]))
        return pairs



