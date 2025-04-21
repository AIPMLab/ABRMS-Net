import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Concatenate,
    GlobalMaxPooling2D,
    Dropout,
    Dense,
    Lambda,
    Reshape,
    Activation,
    Multiply,
    Add,
    UpSampling2D,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# 设置随机种子以确保结果可复现
os.environ['PYTHONHASHSEED'] = '1'
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Mish激活函数
def mish(x):
    return x * K.tanh(K.softplus(x))

def Multiscale_Convolution_Block(inputs, filters):
    a = Conv2D(filters, 3, padding='same')(inputs)
    a = BatchNormalization()(a)
    a = Activation(mish)(a)

    b = Conv2D(filters, 5, padding='same')(inputs)
    b = BatchNormalization()(b)
    b = Activation(mish)(b)

    c = Conv2D(filters, 7, padding='same')(inputs)
    c = BatchNormalization()(c)
    c = Activation(mish)(c)

    d = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    d = Conv2D(filters, 1, padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation(mish)(d)

    mid = Concatenate()([a, b, c, d])
    mid = Conv2D(filters, 1, padding='same')(mid)
    mid = BatchNormalization()(mid)
    mid = Activation(mish)(mid)

    return mid

def min_pooling2d(inputs):
    return -K.pool2d(-inputs, pool_size=(inputs.shape[1], inputs.shape[2]), pool_mode='max')

def Global_attention_block(C_A):
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(C_A)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(C_A)
    min_pool = min_pooling2d(C_A)
    global_avg_pool = GlobalAveragePooling2D(keepdims=True)(C_A)

    upsample_size = C_A.shape[1:3]
    upsampled_min_pool = UpSampling2D(size=upsample_size)(min_pool)
    upsampled_min_pool = Conv2D(1, (1, 1), padding='same')(upsampled_min_pool)
    upsampled_global_avg_pool = UpSampling2D(size=upsample_size)(global_avg_pool)
    upsampled_global_avg_pool = Conv2D(1, (1, 1), padding='same')(upsampled_global_avg_pool)

    concatenated_pools = Concatenate()([avg_pool, max_pool, upsampled_min_pool, upsampled_global_avg_pool])
    relu = Activation(mish)(concatenated_pools)
    conv_sigmoid = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(relu)
    S_A = Multiply()([conv_sigmoid, C_A])

    return S_A

def self_attention_block(inp):
    shp = inp.shape
    a = Conv2D(shp[3] // 8, 1, padding='same')(inp)
    a = Activation(mish)(a)

    b = Conv2D(shp[3] // 8, 1, padding='same')(inp)
    b = Activation(mish)(b)

    c = Conv2D(shp[3] // 8, 1, padding='same')(inp)
    c = Activation(mish)(c)

    a = Reshape((shp[1] * shp[2], shp[3] // 8))(a)
    b = Reshape((shp[1] * shp[2], shp[3] // 8))(b)
    b = K.permute_dimensions(b, (0, 2, 1))
    c = Reshape((shp[1] * shp[2], shp[3] // 8))(c)
    inter = K.batch_dot(a, b)
    inter = Activation('softmax')(inter)
    out = K.batch_dot(inter, c)
    out = Reshape((shp[1], shp[2], shp[3] // 8))(out)
    out = Conv2D(shp[3], 1, padding='same')(out)
    out = Activation(mish)(out)

    return out

def channel_attention(inputs):
    shape = K.int_shape(inputs)
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(shape[3] // 8, activation=mish, kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(shape[3], activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
    x = Multiply()([x, inputs])

    return x

def spatial_attention(inputs):
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(inputs)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(inputs)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    spatial_attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', use_bias=False)(concat)
    return Multiply()([inputs, spatial_attention])

def load_model():
    K.clear_session()
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(32, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(mish)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(mish)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(mish)(x)
    x = MaxPooling2D()(x)

    a1 = Multiscale_Convolution_Block(x, 128)
    a1 = Add()([a1, x])
    a1 = Activation(mish)(a1)

    a12 = Multiscale_Convolution_Block(a1, 128)
    a12 = Add()([a12, a1])
    a12 = Activation(mish)(a12)

    mid1 = Concatenate()([a1, a12])
    mid1 = BatchNormalization()(mid1)
    mid1 = Activation(mish)(mid1)
    a13 = Multiscale_Convolution_Block(mid1, 128)
    mid2 = Concatenate()([a1, a12, a13])
    mid2 = BatchNormalization()(mid2)
    mid2 = Activation(mish)(mid2)
    x = MaxPooling2D()(mid2)

    a2 = Multiscale_Convolution_Block(x, 256)
    x = Conv2D(256, (1, 1), padding='same')(x)
    a2 = Add()([a2, x])
    a2 = Activation(mish)(a2)
    x = MaxPooling2D()(a2)

    a3 = Multiscale_Convolution_Block(x, 512)
    a31 = self_attention_block(a3)
    a32 = Global_attention_block(a3)
    a3 = Add()([a31, a32])
    x = channel_attention(a3)
    x = spatial_attention(x)

    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)

    return model


def custom_objects():
    return {'mish': mish}

model = load_model()
total_params = model.count_params()
print(f"Total number of parameters: {total_params}")