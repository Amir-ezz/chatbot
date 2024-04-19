import numpy as np
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tensorflow as tf

#import tflearn
import random
import json
import pickle
import json
with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Input
    import numpy as np
    import random


    # تعريف bag_of_words function
    def bag_of_words(s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return np.array(bag)


    # تعريف chat function
    def chat():
        print("Start talking with the bot (type quit to stop)!")
        while True:
            inp = input("You: ")
            if inp.lower() == "quit":
                break

            results = model.predict(np.array([bag_of_words(inp, words)]))
            results_index = np.argmax(results)
            tag = labels[results_index]

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))


    if __name__ == "__main__":
        # تحديد مدخل النموذج
        input_shape = len(words),
        input_layer = Input(shape=input_shape)

        # إنشاء Keras model
        model = Sequential([
            input_layer,
            Dense(8, activation='relu'),
            Dense(8, activation='relu'),
            Dense(len(output[0]), activation='softmax')
        ])

        # تجميع النموذج
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # تحميل أو تدريب النموذج
        # تحميل أو تدريب النموذج
        try:
            model.load_weights("model_keras.weights.h5")
        except:
            model.fit(np.array(training), np.array(output), epochs=1000, batch_size=8, verbose=1)
            model.save_weights("model_keras.weights.h5")

        # بدء الدردشة
        chat()
