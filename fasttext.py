import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dot, Activation, Multiply
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Step 1: Load and preprocess the poetry dataset from .csv file
def load_poetry_data(file_path):
    data = pd.read_csv(file_path)
    poetry_lines = data["teks"].tolist()
    return poetry_lines


file_path = "data/pantun.csv"
poetry_lines = load_poetry_data(file_path)

# Step 2: Tokenize the poetry dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poetry_lines)
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(poetry_lines)
sequences = pad_sequences(sequences, padding='post')

# Step 3: Load FastText word vectors
fasttext_vectors_file = "data/id_fasttext.txt"
embeddings_index = {}
with open(fasttext_vectors_file, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Step 4: Create the embedding matrix
embedding_dim = len(embeddings_index['amanah'])  # Adjust this based on your FastText embeddings
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Step 5: Build and train the BiLSTM model with Attention mechanism
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=None, trainable=False))
model.add(Bidirectional(LSTM(128, return_sequences=True)))

# Attention mechanism: Dot product of attention scores and encoder output
def attention_dot(inputs):
    encoder_output, attention_scores = inputs
    context_vector = Dot(axes=1)([attention_scores, encoder_output])
    return context_vector

model.add(Dense(1))  # Attention layer outputting attention scores
model.add(Activation('softmax'))  # Apply softmax to get attention weights
model.add(Dense(vocab_size))  # Output layer for attention scores
model.add(Activation('softmax', name='attention_softmax'))  # Apply softmax to get normalized attention scores

# Compute the context vector using the attention_dot function and attention scores
model.add(Multiply(name='context_mul'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


# Prepare input and output data for the BiLSTM model
X = sequences[:, :-1]  # Input sequences (all words except the last)
y = sequences[:, 1:]   # Output sequences (all words except the first)

# Train the model
model.fit(X, y, batch_size=64, epochs=50)

# Step 6: Generate poetry using the trained BiLSTM model with Beam Search and adjusted temperature
def generate_poetry(seed_text, num_words, beam_width=5, temperature=1.0):
    generated_poems = [seed_text]
    for _ in range(num_words):
        seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
        input_sequence = pad_sequences([seed_sequence], padding='post')
        predicted_weights = model.predict(input_sequence)[0]

        # Apply temperature-based sampling
        predicted_weights = np.log(predicted_weights) / temperature
        exp_weights = np.exp(predicted_weights)
        predicted_probs = exp_weights / np.sum(exp_weights)

        # Beam Search
        best_indices = predicted_probs.argsort()[-beam_width:][::-1]
        next_word_idx = np.random.choice(best_indices, p=predicted_probs[best_indices])

        next_word = tokenizer.index_word.get(next_word_idx, '')
        generated_poems = [poem + " " + next_word for poem in generated_poems]

        # Update seed_text for the next iteration
        seed_text = generated_poems[-1]

    return generated_poems

# Example usage:
seed_text = "Rayuan pulau kelapa"
num_words_to_generate = 20
generated_poems = generate_poetry(seed_text, num_words_to_generate, beam_width=5, temperature=0.7)
for poem in generated_poems:
    print(poem)
