import numpy as np
import pandas as pd

from keras.models import Model


import pyltr
import pickle
import ranking
import building
import training
from tqdm import tqdm
from sklearn.neighbors import KDTree

def extract_embedded_features(model, df, emb, max_len=40):
    intermediate_model = Model(input=model.input,
                            output=model.get_layer(model.layers[5].name).output)
    vecs = building.build4head(building.df2vec(df, emb), max_len)
    return intermediate_model.predict(vecs)


tree = None

def get_best_continues(all_embedded_features,
        base_map_name='datasets/base_map_mid.npy'):
    global tree
    contexsts = all_embedded_features[:, :3 * 40]

    if tree is None:
        base_map = np.load(base_map_name)
        tree = KDTree(base_map[:, 0])

    features = np.zeros((len(contexts), 300))
    for i, c in tqdm(list(enumerate(contexsts)), position=0):
        _, ids= tree.query(c.reshape(1, -1), 10)
        features[i] = np.mean(base_map[ids, 1], axis=0)

    return features



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
    last_embedded_features = extract_embedded_features(net, df, emb)[:, 2 * 40:]
    all_input_embeddings = building.df2vec(df, emb)
    input_embeddings = np.mean(all_input_embeddings.reshape(-1, 4, 40, 300), axis = 2).reshape(-1, 4, 300)
    input_features = np.average(input_embeddings, axis=-2, weights=[1, 2, 4, 4])

    # best_continues = get_best_continues(np.mean(all_input_embeddings[:, :3 * 40], axis=1))
    return np.concatenate([net_pred, last_embedded_features, input_features], axis=1)


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

    rank_model.fit(train_pool, eval_set=val_pool, plot=False,
        logging_level='Verbose')

    if model_file is not None:
        with open(model_file, 'wb') as f:
            pickle.dump([rank_model], f, -1)

    y_score = rank_model.predict(val_pool)
    res = ranking.ndcg(y_val, y_score, groups_val)
    return res, rank_model




def train_classifier(rank_model, net_model, emb, train_df, val_df, model_file=None, use_cache=False):

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



    rank_model.fit(X_train, y_train > 0.5)

    if model_file is not None:
        with open(model_file, 'wb') as f:
            pickle.dump([rank_model], f, -1)

    y_score = rank_model.predict_proba(X_val)[:, 1]
    print(y_score.shape, y_val.shape)
    res = ranking.ndcg(y_val > 0.5, y_score, groups_val)
    return res, rank_model
