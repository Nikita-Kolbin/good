from keras.utils import pad_sequences
from keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
import json


with open('tokenizer.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    tokenizer = tokenizer_from_json(data)
max_sequence_length = 250

model = load_model('model.keras')


def encode(test_input):
    test_input_sequences = tokenizer.texts_to_sequences(test_input)
    test_input_sequences = pad_sequences(test_input_sequences, maxlen=max_sequence_length, padding='post')
    return test_input_sequences


def decode(predicted_output_text):
    result_string = ""
    for index, char in enumerate(predicted_output_text[0]):
        if index % 2 == 0:
            result_string += char
    return result_string


while True:
    s = input("Входная фраза(exit для выхода): ").lower()
    if s == 'exit':
        break
    test_input = [s]
    test_input_sequences = encode(test_input)
    predicted_output_sequences = model.predict(test_input_sequences, verbose=0)
    predicted_output_text = tokenizer.sequences_to_texts(predicted_output_sequences.argmax(axis=-1))
    result_string = decode(predicted_output_text)
    print("Перевод в творительный:", result_string, '\n')
