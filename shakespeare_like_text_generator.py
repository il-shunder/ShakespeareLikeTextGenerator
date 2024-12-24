import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

MODEL_PATH = "shakespeare.keras"
SEQUENCE_LENGTH = 40
STEP = 3

filepath = tf.keras.utils.get_file(
    "shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)

text = open(filepath, "rb").read().decode(encoding="utf-8").lower()
characters = sorted(set(text))

char2index = {char: idx for idx, char in enumerate(characters)}
index2char = {idx: char for idx, char in enumerate(characters)}

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except ValueError:
    sentences = []
    next_chars = []

    for i in range(0, len(text) - SEQUENCE_LENGTH, STEP):
        sentences.append(text[i : i + SEQUENCE_LENGTH])
        next_chars.append(text[i + SEQUENCE_LENGTH])

    x = np.zeros((len(sentences), SEQUENCE_LENGTH, len(characters)), dtype=np.bool)
    y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char2index[char]] = 1
        y[i, char2index[next_chars[i]]] = 1

    timesteps, features = SEQUENCE_LENGTH, len(characters)

    model = Sequential(
        [
            Input((timesteps, features)),
            LSTM(128),
            Dense(len(characters), activation="softmax"),
        ]
    )

    model.compile(loss=CategoricalCrossentropy(), optimizer=RMSprop(learning_rate=0.01))

    model.fit(x, y, epochs=10, batch_size=256)

    model.save(MODEL_PATH)
finally:
    if model:

        def sample_index(preds, temperature):
            preds = np.asarray(preds).astype("float64")
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        def generate_text(length=100, temperature=0.6):
            start_index = random.randint(0, len(text) - SEQUENCE_LENGTH - 1)
            generated = ""
            sentence = text[start_index : start_index + SEQUENCE_LENGTH]
            generated += sentence
            for i in range(length):
                x = np.zeros((1, SEQUENCE_LENGTH, len(characters)), dtype=np.bool)
                for t, char in enumerate(sentence):
                    x[0, t, char2index[char]] = 1

                predictions = model.predict(x, verbose=0)[0]
                next_index = sample_index(predictions, temperature)
                next_char = index2char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char
            return generated

        print("--------0.2--------")
        print(generate_text(300, 0.2))
        print("--------0.4--------")
        print(generate_text(300, 0.4))
        print("--------0.6--------")
        print(generate_text(300, 0.6))
        print("--------0.8--------")
        print(generate_text(300, 0.8))
        print("--------1.0--------")
        print(generate_text(300, 1.0))
