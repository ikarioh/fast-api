import pickle

from functions import *



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



text = "question: way select item cs grid 'grid column change style using javascript dom want select grid item target 'grid column style cs want " \
       "actively change using javascript idea select dom figured could use something like document.queryselector '.classname .style.grid column n't worked"

print(key_words(text))