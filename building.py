import numpy as np
import pandas as pd

import regex
from nltk.stem.snowball import RussianStemmer


def sent2vec(sent, emb, max_len=30, emb_size=300):
    vec = np.zeros((max_len, emb_size))
    for i, t in enumerate(sent[:min(max_len, len(sent))]):
        if t in emb:
            vec[i] = emb[t]
    return vec


def fix_text(text):
    text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = regex.sub("&[a-z]+;", "", text) # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = regex.sub("(?s){.+?}", "", text) # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text) # remove bold symbols
    text = regex.sub("[']{2}", "", text) # remove italic symbols
    text = regex.sub(u"[^ \r\n\p{Cyrillic}.?!\-]", " ", text)
    text = text.lower()
    return text


def tokenize_word(word, stemmer):
    stem =  stemmer.stem(word)
    affix = word[len(stem):]

    if affix:
        return (stem, '#' + affix)
    else:
        return (stem, )


def clear_sent(sent, stemmer=RussianStemmer):
    sent = fix_text(sent)
    phrases = regex.split("([.?!])?[\n]+|[.?!] ", sent)
    words = [s.split() for s in phrases if s is not None]

    sent = []
    for s in words:
        sent.append('<s>')
        for w in s:
            tokens = tokenize_word(w, stemmer)
            for t in tokens:
                sent.append(t)
        sent.append('<\s>')

    return sent


def context2vec(context, emb, max_len=30, emb_size=300, stemmer=RussianStemmer()):
    assert len(context) == 4
    vecs = None
    for sent in context:
        sent = clear_sent(sent, stemmer)
        vec = sent2vec(sent, emb, max_len, emb_size)
        if vecs is None:
            vecs = vec
        else:
            vecs = np.concatenate([vecs, vec], axis = 0)

    return vecs


def build4head(ndvec, max_len):
    assert ndvec.shape[1] == 4 * max_len
    return [ndvec[:, :max_len], ndvec[:, max_len:2*max_len],
        ndvec[:, 2*max_len:3*max_len], ndvec[:, 3*max_len:4*max_len]]


def df2vec(df, emb):
    res = []
    col_set = [1,2,3,5]
    if 1 not in df.columns:
        col_set = [str(i) for i in col_set]

    for i in range(df.shape[0]):
        res.append(context2vec(df[col_set][i:i+1].values[0], emb, max_len=40))
    return np.array(res)


def shuffle_by_groups(df, col, random_state=None):
    groups = [df for _, df in df.groupby(col)]
    np.random.seed(random_state)
    np.random.shuffle(groups)
    return pd.concat(groups).reset_index(drop=True)


def make_emb(file):
    emb_csv = pd.read_csv(file, header=None).drop([301], axis=1)
    return {str(line[0]): np.array(line[1:]) for line in emb_csv.values}
