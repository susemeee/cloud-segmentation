import sys
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras import backend as K

def relu6(x):
  return K.relu(x, max_value=6)

@tf.function
def preprocess_image(image):
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (320, 240,), method=tf.image.ResizeMethod.LANCZOS3)
    return preprocess_input(image)
     
with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}):

    model = load_model(os.path.join(os.getcwd(), 'result', 'resultnet.hdf5'), custom_objects={"tf": tf})

    image = tf.io.read_file(sys.argv[1])
    image = preprocess_image(image)

    pred = model.predict(np.expand_dims(image, 0))
    labels = np.argmax(pred.squeeze(), -1)

    # remove padding and resize back to original image
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((240, 320,)))

    plt.imshow(labels)
    plt.waitforbuttonpress()