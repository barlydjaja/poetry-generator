import numpy as np

# Split input and target data
input_sequences = np.array(input_sequences)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))
