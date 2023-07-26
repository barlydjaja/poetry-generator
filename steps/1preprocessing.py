import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense


# 1. preprocessing
def load_poetry_data(file_path):
    data = pd.read_csv(file_path)
    poetry_lines = data["puisi_with_header"].tolist()
    return poetry_lines


poems = load_poetry_data('../data/puisi_small.csv')  # List of strings containing poems

# for x, line in enumerate(poems):
#     if not isinstance(line, str):
#         print('type: ', type(line))
#         print('something is wrong: ', line)
#         print('index: ', x)
#         print('poems: ', poems[x])
#         print('prev poems', poems[x-1])
# print('done')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(poems)
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in poems:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to have the same length
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')


# 2. Preparing
# Split input and target data
input_sequences = np.array(input_sequences)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))

# print('x', X)
# print('y', y)


# 3. build and train
# Create the Bi-LSTM model
embedding_size = 100
model = Sequential()
model.add(Embedding(total_words, embedding_size, input_length=max_sequence_length-1))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=100, verbose=1)

print('model', model)


# 4. generate the poetry
# Function to generate poetry
def generate_poetry(seed_text, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_word_index = np.argmax(model.predict(token_list), axis=-1)
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text += " " + predicted_word
    return seed_text

# Generate poetry
seed_text = "Pulau Kelapa"
num_words_to_generate = 50
generated_poetry = generate_poetry(seed_text, num_words_to_generate)
print(generated_poetry)


