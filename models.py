import keras.backend as K
from keras.layers import Input, Conv1D, MaxPool1D, UpSampling1D, Flatten, \
    LeakyReLU, BatchNormalization, Dense, Dropout, concatenate, Reshape, Lambda
from keras.activations import sigmoid
from keras.regularizers import l2
from keras.models import Model
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import Adadelta, Adam


def model(input_shape, eps = 1e-4):
    context1_input = Input(shape=(input_shape), name='context1')
    context2_input = Input(shape=(input_shape), name='context2')
    context3_input = Input(shape=(input_shape), name='context3')
    context4_input = Input(shape=(input_shape), name='context4')

    conv_inputs = Input(shape=(input_shape), name='conv_input')
    conv_net = Conv1D(filters=128, kernel_size=10, padding='same',
        kernel_regularizer=l2(eps), bias_regularizer=l2(eps))(conv_inputs)
    conv_net = BatchNormalization()(conv_net)
    conv_net = LeakyReLU(alpha=0.05)(conv_net)
    conv_net = Dropout(0.1)(conv_net)
    conv_net = MaxPool1D(pool_size=40)(conv_net)

#     conv_net = Conv1D(filters=256, kernel_size=10, padding='same',
#         kernel_regularizer=l2(eps), bias_regularizer=l2(eps))(conv_net)
#     conv_net = BatchNormalization()(conv_net)
#     conv_net = LeakyReLU(alpha=0.05)(conv_net)
#     conv_net = Dropout(0.1)(conv_net)
#     conv_net = MaxPool1D(pool_size=8)(conv_net)

    dense_net = Flatten()(conv_net)
    dense_net = Dense(256, kernel_regularizer=l2(eps),
                    bias_regularizer=l2(eps))(dense_net)
    dense_out = LeakyReLU(alpha=0.1)(dense_net)
    shared_model = Model(inputs=conv_inputs, outputs=dense_out)

    emb1 = shared_model(context1_input)
    emb2 = shared_model(context2_input)
    emb3 = shared_model(context3_input)
    emb4 = shared_model(context4_input)

    emb_input = Input(shape=(256,), name='emb_input')
    deconv_net = Dense(128, kernel_regularizer=l2(eps),
                    bias_regularizer=l2(eps))(emb_input)
    deconv_net = LeakyReLU(alpha=0.1)(deconv_net)
    deconv_net = Reshape((1, 128))(deconv_net)
#     deconv_net = UpSampling1D(size=8)(deconv_net)
#     deconv_net = Conv1D(filters=128, kernel_size=10, padding='same',
#         kernel_regularizer=l2(eps), bias_regularizer=l2(eps))(deconv_net)
    deconv_net = UpSampling1D(size=40)(deconv_net)
    deconv_net = Conv1D(filters=300, kernel_size=10, padding='same',
        kernel_regularizer=l2(eps), bias_regularizer=l2(eps))(deconv_net)
    deconv_model = Model(inputs=emb_input, outputs=deconv_net)
    
    auto3_net = deconv_model(emb3)
    auto4_net = deconv_model(emb4)
    
    auto3_out = Lambda(lambda x:x,name='auto3')(auto3_net)
    auto4_out = Lambda(lambda x:x,name='auto4')(auto4_net)

    mutual_net = concatenate([emb1, emb2, emb3, emb4])
#     mutual_net = Dense(512, kernel_regularizer=l2(eps),
#                     bias_regularizer=l2(eps))(mutual_net)
#     mutual_net = LeakyReLU(alpha=0.05)(mutual_net)
#     mutual_net = Dropout(0.1)(mutual_net)
    mutual_net = Dense(512, kernel_regularizer=l2(eps),
                    bias_regularizer=l2(eps))(mutual_net)
    mutual_net = LeakyReLU(alpha=0.05)(mutual_net)
    mutual_net = Dropout(0.1)(mutual_net)

    output = Dense(units=1,  kernel_regularizer=l2(eps),
                    bias_regularizer=l2(eps), activation=sigmoid, name='class_out')(mutual_net)

    model = Model(inputs=[context1_input, context2_input,
                            context3_input, context4_input], 
                  outputs=[output, auto3_out, auto4_out])

    model.compile(
            loss={'class_out': binary_crossentropy, 
                  'auto3' : mean_squared_error, 
                  'auto4': mean_squared_error},
            loss_weights = {'class_out': 1, 
                  'auto3' : 1, 
                  'auto4': 1},
            optimizer='rmsprop',
    )

    return model
