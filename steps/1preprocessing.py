import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
import pickle


# 1. preprocessing
def load_poetry_data(file_path):
    data = pd.read_csv(file_path)
    poetry_lines = data["puisi_with_header"].tolist()
    return poetry_lines


poems = load_poetry_data('../data/puisi.csv')  # List of strings containing poems

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
embedding_size = 300

fasttext_vectors_file = "../data/id_fasttext.txt"
embeddings_index = {}
with open(fasttext_vectors_file, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Step 4: Create the embedding matrix
embedding_dim = len(embeddings_index['pribadi'])  # Adjust this based on your FastText embeddings
embedding_matrix = np.zeros((total_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# Create the Bi-LSTM model
model = Sequential()
model.add(Embedding(total_words, embedding_size, input_length=max_sequence_length-1, weights=[embedding_matrix]))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
# TODO: find the ideal epochs number
model.fit(X, y, epochs=100, verbose=1)

print('model', model)
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))


# Function to sample from the model's output with a given temperature
def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)



# 4. generate the poetry
# Function to generate poetry
def generate_poetry(seed_text, num_words, temperature=1.0):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list)[0]
        next_word_index = sample_with_temperature(predicted_probs, temperature)
        next_word = tokenizer.index_word[next_word_index]
        # predicted_word_index = np.argmax(model.predict(token_list), axis=-1)
        # predicted_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text += " " + next_word
    return seed_text

# Generate poetry
seed_text = "Api Membara"
num_words_to_generate = 50
temperature = 0.7
generated_poetry = generate_poetry(seed_text, num_words_to_generate, temperature)
print(generated_poetry)


