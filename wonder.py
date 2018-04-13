import numpy as np
import pandas as pd


def create_sub(df, y_pred, fname):
    df['pred'] = y_pred.reshape(-1)
    groups = df.groupby(by='0')['pred'].apply(lambda x: np.argsort(x)[::-1])
    groups = groups.reset_index()[['0', 'pred']]
    groups.to_csv(fname, index=False, sep=' ', header=False)
    return groups
