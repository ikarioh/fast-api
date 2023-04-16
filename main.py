from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn


import pickle

from functions import *

import tensorflow as tf
import tensorflow_hub as hub
from transformers import TFGenerationMixin


# Bert
import os


os.environ["TF_KERAS"]='1'

# Define the API endpoint
app = FastAPI()


class Sentence(BaseModel):
    question: str

# Load the model
'''loaded_module = tf.saved_model.load('./use')
loaded_embed = loaded_module.signatures['serving_default']

with open('use_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('use_xgboost_model.pkl', 'rb') as f:
    xgb_clf = pickle.load(f)'''

with open('lda_id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)

with open('lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

def transform_num(txt):
    tt = transform_lem_sentence_fct(txt)
    return id2word.doc2bow(tt)


def best_topics(min_proba=0.1, lda_topics=[]):
    return sorted(list(filter(lambda p: p[1] > min_proba, [c for c in lda_topics])), key=lambda v: v[1], reverse=True)


def key_words(text, min_prob_topics=0.1, min_prob_words=0.015):
    text_1 = transform_bow_lem_fct(text)

    text_num = transform_num(text_1)

    text_lda = lda_model[text_num]

    best_2_topics = [t[0] for t in best_topics(min_prob_topics, text_lda[0])][:2]

    return best_2_topics, {t: list(filter(lambda p: p[1] > min_prob_words, lda_model.show_topic(t))) for t in best_2_topics}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/Words_proposition")
async def propose(request: Sentence):

    #print(request.question)
    # Return the prediction
    topics = key_words(request.question)

    return {number: [w[0] for w in words] for number, words in topics[1].items()}

'''@app.post("/Tags_prediction")
async def predict(request: Sentence):

    tab = ['javascript', 'java', 'c#', 'python', 'php', 'android', 'c++', 'html', 'jquery', 'css']

    print(request.question)

    sentence = transform_dl_fct(request.question)

    vecteur = loaded_module([sentence])

    vecteur_scaled = scaler.transform(vecteur.numpy())

    y_pred = np.array(xgb_clf.predict(vecteur_scaled)[0])

    print(y_pred)

    results = [val for val in np.array(tab)[np.argwhere(y_pred==1)][0]]


    #index = [i for i in range(len(y_pred[0])) if y_pred[0][i] == 1]

    #y_pred = [tab[i] for i in index]

    # Return the prediction
    return {"tags": results}'''


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


# uvicorn main:app --reload
