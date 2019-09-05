import os
import tensorflow as tf
from deeplabv3.model import Deeplabv3, relu6
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import multi_gpu_model

from run_dataset import train_dataset, val_dataset, BATCH_SIZE

# mirrored_strategy = tf.distribute.MirroredStrategy()
# https://github.com/tensorflow/tensorflow/issues/21470#issuecomment-422506263
mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


tensorboard_logger = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('result', 'logdir'), histogram_freq=10, batch_size=BATCH_SIZE, write_graph=True, write_grads=True, write_images=True, update_freq='epoch')
csv_logger = CSVLogger('./result/logs/logfile.txt')
checkpointer = ModelCheckpoint(filepath='./result/cloudsegnet.hdf5', save_best_only=True, monitor='val_loss')

with mirrored_strategy.scope():
    deeplab_model = Deeplabv3(weights=None, input_shape=(320, 240, 3), classes=2, activation='sigmoid')
    model = multi_gpu_model(deeplab_model, gpus=2)
    model_path = os.path.join(os.getcwd(), 'result', 'cloudsegnet.hdf5')
    if os.path.isfile(model_path):
        model.load_weights(model_path)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.fit(
      train_dataset, epochs=400,
      validation_data=val_dataset,
      steps_per_epoch=1920 / BATCH_SIZE,
      validation_steps=1920 * 0.2 / BATCH_SIZE,
      verbose=1, callbacks=[csv_logger, checkpointer, tensorboard_logger])

# eval
print('\n# Evaluate')
model.save(os.path.join(os.getcwd(), 'result', 'cloudsegnet.model'))
