# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 21:20:07 2023

@author: shubh
"""

import tensorflow as tf
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')
import numpy as np
from matplotlib import pyplot as plt

import cv2
import imghdr

data_dir='F:/Project 2/spectograms/Train test/Train/'

image_exts = ['png']

image_exts[0]

'''
for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
print(image)
'''

data = tf.keras.utils.image_dataset_from_directory(data_dir)


data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

batch[0].shape 

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

data = data.map(lambda x,y: (x/255, y))

data.as_numpy_iterator().next()[0].max()

scaled=batch[0]/255
scaled.min()

len(data)

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(128, (3,3), 1, activation='relu', input_shape=(256,256,1)))
model.add(MaxPooling2D())
model.add(Conv2D(256, (3,3), 1, activation='relu')) #does not activate all the neurons at the same time
model.add(MaxPooling2D())
model.add(Conv2D(512, (3,3), 1, activation='relu')) #Conv 2D taking the maximum value over an input window
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))



#model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy']) #logloss metrics

model.compile('adam', loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy']) #logloss metrics

model.summary()

logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=5, validation_data=val, callbacks=[tensorboard_callback])



fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy #MultiAccuracy,
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result(), re.result(), acc.result())

img = cv2.imread("F:/Project 2/spectograms/Blank Folders/1/153851.png")
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat[0])



from tensorflow.keras.models import load_model
model.save(os.path.join('models','imageclassifier.h5'))
new_model = load_model('imageclassifier.h5')
new_model.predict(np.expand_dims(resize/255, 0))
#array([[0.01972741]], dtype=float32)

