import os
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# image path to array
dataset_path = os.path.join(os.getcwd(), '..', '__datasets__')
cloud_path = os.path.join(dataset_path, 'cloud_gt', 'images')
gt_path = os.path.join(dataset_path, 'cloud_gt', 'masks')

# shuffling
SEED = 1234
BATCH_SIZE = 16
# 1600
BUFFER_SIZE = 100

genargs = dict(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    rotation_range=60,
    shear_range=60,
    width_shift_range=0.01,
    height_shift_range=0.01,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    channel_shift_range=0,
    fill_mode='reflect'
)

imagegen = ImageDataGenerator(**genargs, brightness_range=(0.975, 1.2,))
maskgen = ImageDataGenerator(**genargs)

# SEE: Keras has an issue https://github.com/keras-team/keras-preprocessing/issues/236
K__BATCH_SIZE = 1
image_generator = imagegen.flow_from_directory(cloud_path, shuffle=False, target_size=(320,240,),
                                              class_mode=None, seed=SEED, batch_size=K__BATCH_SIZE,
                                              subset='training')
mask_generator = maskgen.flow_from_directory(gt_path, shuffle=False, target_size=(320,240,),
                                              class_mode=None, seed=SEED, batch_size=K__BATCH_SIZE,
                                              subset='training', color_mode='grayscale')
image_val_generator = imagegen.flow_from_directory(cloud_path, shuffle=False, target_size=(320,240,),
                                              class_mode=None, seed=SEED, batch_size=K__BATCH_SIZE,
                                              subset='validation')
mask_val_generator = maskgen.flow_from_directory(gt_path, shuffle=False, target_size=(320,240,),
                                              class_mode=None, seed=SEED, batch_size=K__BATCH_SIZE,
                                              subset='validation', color_mode='grayscale')

image_dataset = tf.data.Dataset.from_generator(lambda: image_generator, tf.float32, tf.TensorShape([K__BATCH_SIZE, 320, 240, 3])).map(lambda t: tf.reshape(t, [320, 240, 3]))
mask_dataset = tf.data.Dataset.from_generator(lambda: mask_generator, tf.float32, tf.TensorShape([K__BATCH_SIZE, 320, 240, 1])).map(lambda t: tf.reshape(t, [320, 240, 1]))
image_val_dataset = tf.data.Dataset.from_generator(lambda: image_val_generator, tf.float32, tf.TensorShape([K__BATCH_SIZE, 320, 240, 3])).map(lambda t: tf.reshape(t, [320, 240, 3]))
mask_val_dataset = tf.data.Dataset.from_generator(lambda: mask_val_generator, tf.float32, tf.TensorShape([K__BATCH_SIZE, 320, 240, 1])).map(lambda t: tf.reshape(t, [320, 240, 1]))

train_dataset = tf.data.Dataset.zip((image_dataset, mask_dataset,)).batch(BATCH_SIZE).shuffle(BUFFER_SIZE)
val_dataset = tf.data.Dataset.zip((image_val_dataset, mask_val_dataset,)).batch(BATCH_SIZE).shuffle(BUFFER_SIZE)

if __name__ == '__main__':
    logdir = os.path.join('result', 'logdir', 'train_data', datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        i = 0
        # Don't forget to reshape.
        BATCH_SIZE = 1
        for x, y in train_dataset:
            tf.summary.image("{} training data examples".format(BATCH_SIZE), x, max_outputs=BATCH_SIZE, step=0)
            tf.summary.image("{} mask data examples".format(BATCH_SIZE), y, max_outputs=BATCH_SIZE, step=0)
            if i >= 50:
                break
            i += 1
