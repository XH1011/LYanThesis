from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
from scipy import linalg
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_data(file,BATCH_SIZE):
    f = open(file, 'rb')
    x0=pickle.load(f)
    x = tf.data.Dataset.from_tensor_slices(x0)
    x = x.shuffle(5000).batch(BATCH_SIZE)
    return x,x0

def load_data2(file,BATCH_SIZE):
    f = open(file, 'rb')
    x0=pickle.load(f)[0]
    x = tf.data.Dataset.from_tensor_slices(x0)
    x = x.shuffle(5000).batch(BATCH_SIZE)
    return x,x0

def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=(1, 2), interpolation=interpolation)(x)
        x = layers.Conv2D(
            width, kernel_size=5, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply