import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

""" TD cao khi có nhiều từ unique (chỉ xuất hiện 1 lần) trong danh sách top words của tất cả các topic. 
TD thấp khi có nhiều từ xuất hiện lặp lại trong top words của nhiều topic
"""

def compute_TD(texts):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
    counter = vectorizer.fit_transform(texts).toarray()

    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)

    return TD


def compute_topic_diversity(top_words, _type="TD"):
    TD = compute_TD(top_words)
    return TD
