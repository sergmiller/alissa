import keras.backend as K
from keras.layers import Input, Conv1D, MaxPool1D, UpSampling1D, Flatten, \
    LeakyReLU, BatchNormalization, Dense, Dropout, concatenate
from keras.activations import sigmoid
from keras.regularizers import l2
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.optimizers import Adadelta


def model1(input_shape, eps = 1e-5):
    '''
        simple 1D conv model with single outut
    '''
    context1_input = Input(shape=(input_shape), name='context1')
    context2_input = Input(shape=(input_shape), name='context2')
    context3_input = Input(shape=(input_shape), name='context3')
    context4_input = Input(shape=(input_shape), name='context4')

    conv_inputs = Input(shape=(input_shape), name='conv_input')
    conv_net = Conv1D(filters=32, kernel_size=10, padding='same',
        kernel_regularizer=l2(eps), bias_regularizer=l2(eps))(conv_inputs)
    conv_net = BatchNormalization()(conv_net)
    conv_net = LeakyReLU(alpha=0.05)(conv_net)
    conv_net = MaxPool1D(pool_size=1)(conv_net)
#     conv_net = Conv1D(filters=64, kernel_size=10, padding='same')
#     conv_net = BatchNormalization()(conv_net)
#     conv_net = LeakyReLU(alpha=0.05)(conv_net)
#     conv_net = MaxPool1D(pool_size=1)(conv_net)
    dense_net = Flatten()(conv_net)
    dense_net = Dense(128, kernel_regularizer=l2(eps),
                    bias_regularizer=l2(eps))(dense_net)
    dense_out = LeakyReLU(alpha=0.05)(dense_net)
    shared_model = Model(inputs=conv_inputs, outputs=dense_out)

    emb1 = shared_model(context1_input)
    emb2 = shared_model(context2_input)
    emb3 = shared_model(context3_input)
    emb4 = shared_model(context4_input)

    mutual_net = concatenate([emb1, emb2, emb3, emb4])
    mutual_net = Dense(512, kernel_regularizer=l2(eps),
                    bias_regularizer=l2(eps))(mutual_net)
    mutual_net = LeakyReLU(alpha=0.05)(mutual_net)
    mutual_net = Dropout(0.05)(mutual_net)
    mutual_net = Dense(256, kernel_regularizer=l2(eps),
                    bias_regularizer=l2(eps))(mutual_net)

    output = Dense(units=1,  kernel_regularizer=l2(eps),
                    bias_regularizer=l2(eps), activation=sigmoid)(mutual_net)

    model = Model(inputs=[context1_input, context2_input,
                            context3_input, context4_input], outputs=output)

    model.compile(
            loss=binary_crossentropy,
            optimizer=Adadelta(),
    )

    return model
