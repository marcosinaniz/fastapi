import os
import json

import h5py
import numpy as np
import pickle as pk
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

def prepare_img_224(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def prepare_flat(img_224):
    base_model = load_model('vgg16.h5')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    feature = model.predict(img_224)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    return flat

# Loading  valid categories for identifying cars using VGG16
with open('cat_counter.pk', 'rb') as f:
    cat_counter = pk.load(f)

# shortlisting top 27 Categories that VGG16 stores for cars (Can be altered for less or more)
cat_list  = [k for k, v in cat_counter.most_common()[:27]]

global graph
graph = tf.compat.v1.get_default_graph()

CLASS_INDEX_PATH = 'imagenet_class_index.json'

def get_predictions(preds, top=5):

    global CLASS_INDEX
    CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def car_damage_check(img_flat):
    second_check = pk.load(open('second_check.pickle', 'rb')) #damaged vs whole - trained model
    print ("Validating that damage exists...")
    train_labels = ['00-damage', '01-whole']
    preds = second_check.predict(img_flat)
    prediction = train_labels[preds[0]]

    if train_labels[preds[0]] == '00-damage':
        print ("Validation complete - proceeding to location and severity determination")
        print ("\n")
        return True
    else:
        return False