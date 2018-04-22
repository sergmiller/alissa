import numpy as np
import pandas as pd

from ranking import ndcg
import generator


def flow(df, emb, batch_size, y=None, weights=None, shuffle=True, max_len=40):
    ranges = np.arange(df.shape[0])
    df.index = ranges
    assert batch_size <= len(ranges)
    while True:
        for i in np.arange(0, len(ranges) - batch_size + 1, batch_size):
            inds = ranges[i:i + batch_size]
            df_batch = df.iloc[inds]
            x_batch = generator.gen_x(df_batch, emb, max_len)
            batch = [x_batch,]
            if y is not None:
                batch.append([y[inds], x_batch[2], x_batch[3]])
            if weights is not None:
                batch.append([weights[inds], np.ones(inds.shape), np.ones(inds.shape)])
            yield tuple(batch)




def predict_with_nn(model, df, emb, max_len=40, use_y=False):
    X = generator.gen_x(df, emb, max_len)
    groups = df['0'].values

    y_pred = model.predict(X)[0]
    if use_y:
        y_true = generator.gen_y(df)
        score, std = ndcg(y_true, y_pred, groups)
        print('ndcg_mean: {}, ndcg_std: {}'.format(score, std))

    return y_pred


def predict_with_stack_model(rank_model, net_model, df, emb, max_len=40, use_y=False, use_proba=False):
    if use_y:
        X, y = generator.make_dataset(df, net_model, emb)
    else:
        X = generator.make_dataset(df, net_model, emb, False)

    groups = df['0'].values

    if use_proba:
        y_pred = rank_model.predict_proba(X)[:, 1]
    else:
        y_pred = rank_model.predict(X)

    if use_y:
        score, std = ndcg(y, y_pred, groups)
        print('ndcg_mean: {}, ndcg_std: {}'.format(score, std))

    return y_pred
