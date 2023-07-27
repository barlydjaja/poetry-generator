import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random

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

filename = 'trained_model.sav'
model = pickle.load(open('./trained_model.sav', 'rb'))

# 4. generate the poetry
# Function to generate poetry
# def generate_poetry(seed_text, num_words):
#     for _ in range(num_words):
#         token_list = tokenizer.texts_to_sequences([seed_text])[0]
#         token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
#         predicted_word_index = np.argmax(model.predict(token_list), axis=-1)
#         predicted_word = tokenizer.index_word[predicted_word_index[0]]
#         seed_text += " " + predicted_word
#     return seed_text
#
# # Generate poetry
# seed_text = "pulau kelapa"
# num_words_to_generate = 50
# generated_poetry = generate_poetry(seed_text, num_words_to_generate)
# print(generated_poetry)

# Function to sample from the model's output with a given temperature
def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

# Function to generate poetry iteratively with temperature sampling
def generate_poetry_iterative(seed_text, max_words_to_generate=100, temperature=1.0):
    generated_poetry = seed_text

    # Generate poetry iteratively
    for _ in range(max_words_to_generate):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list)[0]
        next_word_index = sample_with_temperature(predicted_probs, temperature)
        next_word = tokenizer.index_word[next_word_index]
        generated_poetry += " " + next_word

        # Check if the generated poetry ends with a punctuation mark or reaches a maximum length
        if next_word in ['.', '!', '?'] or len(generated_poetry.split()) >= max_words_to_generate:
            break

        seed_text = generated_poetry  # Update seed text for the next iteration

    return generated_poetry

# Generate poetry iteratively with temperature sampling
seed_text = "SENJA DI PELABUHAN BESAR"
max_words_to_generate = 70
temperature = 0.7
generated_poetry = generate_poetry_iterative(seed_text, max_words_to_generate, temperature)
print(generated_poetry)


