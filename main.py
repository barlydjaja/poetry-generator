import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dot, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load and preprocess the poetry dataset from .csv file
def load_poetry_data(file_path):
    data = pd.read_csv(file_path)
    poetry_lines = data["puisi"].tolist()
    return poetry_lines

file_path = "data/puisi.csv"
poetry_lines = load_poetry_data(file_path)

# Step 2: Tokenize the poetry dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poetry_lines)
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(poetry_lines)
sequences = pad_sequences(sequences, padding='post')

# Step 3: Build and train the BiLSTM model with Attention mechanism
embedding_dim = 100  # Adjust this based on the desired dimension of word embeddings

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W_q = self.add_weight(name="W_q", shape=(input_shape[-1], input_shape[-1]), initializer="uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        q = tf.matmul(x, self.W_q)
        v = tf.reduce_sum(q, axis=-1, keepdims=True)
        attention_weights = tf.nn.softmax(v, axis=1)
        return tf.reduce_sum(x * attention_weights, axis=1)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=None))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(AttentionLayer())
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Prepare input and output data for the BiLSTM model
X = sequences[:, :-1]  # Input sequences (all words except the last)
y = sequences[:, 1:]   # Output sequences (all words except the first)

# Make sure y contains integer-encoded labels, not one-hot encoded
y = np.argmax(y, axis=-1)

# Train the model
model.fit(X, y, batch_size=64, epochs=50)

# Step 4: Generate poetry using the trained BiLSTM model with Beam Search
def generate_poetry(seed_text, num_words, beam_width=5, temperature=1.0):
    generated_poems = [seed_text]
    for _ in range(num_words):
        seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
        input_sequence = pad_sequences([seed_sequence], padding='post')
        predicted_weights = model.predict(input_sequence)[0]

        # Apply temperature-based sampling
        predicted_weights = np.log(predicted_weights) / temperature
        exp_weights = np.exp(predicted_weights)
        predicted_weights = exp_weights / np.sum(exp_weights)

        # Beam Search
        best_indices = predicted_weights.argsort()[-beam_width:][::-1]
        next_word_idx = np.random.choice(best_indices, p=predicted_weights[best_indices])

        next_word = tokenizer.index_word.get(next_word_idx, '')
        generated_poems = [poem + " " + next_word for poem in generated_poems]

    return generated_poems

# Example usage:
seed_text = "Rayuan pulau kelapa"
num_words_to_generate = 20
generated_poems = generate_poetry(seed_text, num_words_to_generate, beam_width=5, temperature=0.7)
for poem in generated_poems:
    print(poem)
