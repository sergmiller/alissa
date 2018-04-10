import os
import keras


class TelegramCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        os.system("telegram-send 'Start NN training'")

    def on_train_end(self, logs={}):
        os.system("telegram-send 'End NN training'")

    def on_epoch_begin(self, epoch, logs={}):
        os.system("telegram-send 'Epoch {}'".format(epoch + 1))

    def on_epoch_end(self, epoch, logs={}):
        os.system("telegram-send 'loss: {:.4f}, val_loss: {:.4f}'".format(
            logs.get('loss'), logs.get('val_loss')))

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


from sklearn.metrics import roc_auc_score
from building import build4head
from building import df2vec
class RocCallback(keras.callbacks.Callback):
    def __init__(self, train, val, emb, max_len=40):
        self.x =  build4head(df2vec(train, emb), max_len)
        self.x_val = build4head(df2vec(val, emb), max_len)
        self.y = (train['6'] == 'good').values
        self.y_val = (val['6'] == 'good').values

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val, batch_size=128)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        s = "roc-auc: {:.4f} - roc-auc_val: {:.4f}".format(round(roc,4), round(roc_val,4))
        os.system("telegram-send '{}'".format(s))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
