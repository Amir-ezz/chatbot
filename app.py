import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tensorflow as tf
import random
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# تحميل بيانات الانتنت
with open('intents.json') as file:
    data = json.load(file)

# تحميل بيانات التدريب
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# تحميل النموذج
model = Sequential([
    Input(shape=(len(words),)),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(len(output[0]), activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.load_weights("model_keras.weights.h5")

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
def chat(message):
    results = model.predict(np.array([bag_of_words(message, words)]))
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)

# تحديد العنوان والرسالة الافتراضية
st.title("Chatbot")
default_message = "Type a message..."

# مربع النص لإدخال الرسائل
user_input = st.text_input("You:", default_message)

# عرض رسالة الدردشة
if user_input != default_message:
    bot_response = chat(user_input)
    st.write("Bot:", bot_response)
