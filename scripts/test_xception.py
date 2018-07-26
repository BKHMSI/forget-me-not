import os
import keras
from keras.layers import *
from keras.models import Model
import requests
import h5py as h5

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

xc_ds = h5.File('/home/balkhamissi/projects/data/LSMDC/keras_xc-1.h5', 'r')
model = keras.applications.Xception(
            include_top=True, pooling='avg')

feat = xc_ds['0009_Forrest_Gump_00.50.47.276-00.50.48.887.avi'][4]

l_input =  Input(shape=feat.shape)
layers = model.layers[-1](l_input)

model = Model(input=[l_input], outputs=[layers])
predict = model.predict(feat.reshape(1, -1))

print(predict.argmax(axis=1))

print(type(layers))

labels = {int(key):value for (key, value)
            in requests.get(LABELS_URL).json().items()}

print(labels[predict.argmax(axis=1)[0]])