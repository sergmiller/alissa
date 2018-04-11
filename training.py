import numpy as np
import pandas as pd

from building import df2vec, build4head


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
