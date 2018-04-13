import numpy as np
import pandas as pd

from building import df2vec, build4head
from callbacks import fix_val, ndcg


def flow(df, emb, batch_size, y=None, weights=None, shuffle=True, max_len=40):
    ranges = np.arange(df.shape[0])
    df.index = ranges
    assert batch_size <= len(ranges)
    while True:
        for i in np.arange(0, len(ranges) - batch_size + 1, batch_size):
            inds = ranges[i:i + batch_size]
            df_batch = df.iloc[inds]
            x_batch = build4head(df2vec(df_batch, emb), max_len)
            batch = [x_batch,]
            if y is not None:
                batch.append([y[inds], x_batch[2], x_batch[3]])
            if weights is not None:
                batch.append([weights[inds], np.ones(inds.shape), np.ones(inds.shape)])
            yield tuple(batch)




def predict(model, df, emb, max_len=40, use_y=False):
    X =  build4head(df2vec(df, emb), max_len)
    groups = df['0'].values

    y_pred = model.predict(X)[0]
    if use_y:
        y_true = np.array([fix_val(s) for s in df['6'].values])
        score, std = ndcg(y_true, y_pred, groups)
        print('ndcg_mean: {}, ndcg_std: {}'.format(score, std))

    return y_pred
