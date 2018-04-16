import numpy as np
import pandas as pd

from keras.models import Model


import pyltr
import pickle
import ranking
import building
import training

def extract_embedded_features(model, df, emb, max_len=40):
    intermediate_model = Model(input=model.input,
                            output=model.get_layer(model.layers[5].name).output)
    vecs = building.build4head(building.df2vec(df, emb), max_len)
    return intermediate_model.predict(vecs)


def gen_x(df, emb, max_len=40):
    return building.build4head(building.df2vec(df, emb), max_len)

def fix_val(s):
    d = {'good':2, 'neutral':1, 'bad':0}
    return d[s]

def gen_y(df):
    return np.array([fix_val(s) for s in df['6'].values])

from copy import copy
def generate_all_features(df, net, emb):
    net_pred = training.predict_with_nn(net, df, emb, use_y=False)
    embedded_features = extract_embedded_features(net, df, emb)
    input_embeddings = np.mean(building.df2vec(df, emb).reshape(-1, 4, 40, 300), axis = 2).reshape(-1, 4, 300)
    input_features = np.average(input_embeddings, axis=-2, weights=[1, 2, 4, 4])
    return np.concatenate([net_pred, embedded_features, input_features], axis=1)


def make_dataset(df, net, emb, need_y=True):
    X = generate_all_features(df, net, emb)
    if need_y:
        return X, gen_y(df)
    else:
        return X


def train_pyltr(rank_model, metric, net_model, emb, df, test_size=0.1, model_file=None, use_cache=False, stop_after=200):

    df_shuffled = building.shuffle_by_groups(df, '0', random_state=127)
    N = df_shuffled.shape[0]

    inds = np.arange(int(N * test_size))

    val_df = df_shuffled.iloc[inds]
    train_df = df_shuffled.drop(inds)

    if use_cache:
        X_train = np.load('datasets/X_train_boost.npy')
        X_val = np.load('datasets/X_val_boost.npy')
        y_train = np.load('datasets/y_train_boost.npy')
        y_val = np.load('datasets/y_val_boost.npy')
        print('Datasets are loaded')
    else:
        X_train, y_train = make_dataset(train_df, net_model, emb)
        X_val, y_val = make_dataset(val_df, net_model, emb)

        np.save('datasets/X_train_boost', X_train)
        np.save('datasets/X_val_boost', X_val)
        np.save('datasets/y_train_boost', y_train)
        np.save('datasets/y_val_boost', y_val)

        print('Datasets are saved')

    groups_train = train_df['0'].values
    groups_val = val_df['0'].values


    monitor = pyltr.models.monitors.ValidationMonitor(
        X_val, y_val, groups_val, metric=metric, stop_after=stop_after)

    rank_model.fit(X_train, y_train, groups_train, monitor=monitor)

    if model_file is not None:
        with open(model_file, 'wb') as f:
            pickle.dump([rank_model], f, -1)

    y_score = rank_model.predict(X_val)
    res = ranking.ndcg(y_val, y_score, groups_val)
    return res, rank_model


import catboost
def train_catboost(rank_model, net_model, emb, train_df, val_df, model_file=None, use_cache=False):

    if use_cache:
        X_train = np.load('datasets/X_train_boost.npy')
        X_val = np.load('datasets/X_val_boost.npy')
        y_train = np.load('datasets/y_train_boost.npy')
        y_val = np.load('datasets/y_val_boost.npy')
        print('Datasets are loaded')
    else:
        X_train, y_train = make_dataset(train_df, net_model, emb)
        X_val, y_val = make_dataset(val_df, net_model, emb)

        np.save('datasets/X_train_boost', X_train)
        np.save('datasets/X_val_boost', X_val)
        np.save('datasets/y_train_boost', y_train)
        np.save('datasets/y_val_boost', y_val)

        print('Datasets are saved')

    groups_train = train_df['0'].values % int(1e9 + 7)
    groups_val = val_df['0'].values % int(1e9 + 7)

    train_pool = catboost.Pool(X_train, y_train * 0.5, group_id=groups_train.reshape(-1)
                                            ,weight=train_df['7'].values.reshape(-1))

    val_pool = catboost.Pool(X_val, y_val * 0.5, group_id=groups_val.reshape(-1))

    rank_model.fit(train_pool, eval_set=val_pool, plot=True,
        logging_level='Silent')

    if model_file is not None:
        with open(model_file, 'wb') as f:
            pickle.dump([rank_model], f, -1)

    y_score = rank_model.predict(val_pool)
    res = ranking.ndcg(y_val, y_score, groups_val)
    return res, rank_model
