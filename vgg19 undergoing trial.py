# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 22:31:54 2023

@author: shubh
"""

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Set up paths for the train, validation, and test datasets
train_path = 'F:/Project 2/spectograms/Train test/Train/'
val_path = 'F:/Project 2/spectograms/Train test/Val/'
test_path = 'F:/Project 2/spectograms/Train test/Test/'

# Set up the image dimensions and number of classes
img_width, img_height = 224, 224
num_classes = 3

# Set up the VGG19 model with pretrained weights from ImageNet
vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))#(RGB)
#include_top=False - exclude the top (fully connected) layers of the VGG19 model.

# Freeze the layers in VGG19 so that we only train the classifier
for layer in vgg19.layers:
    layer.trainable = False #freezing the weights of these layers.

# Add our own classifier on top of the frozen VGG19 layers
model = Sequential() #simply a linear stack of layers.
model.add(vgg19)
model.add(Flatten()) 
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Set up data augmentation for the train set
#helps in the convergence of the optimization algorithm during the training process.

train_datagen = ImageDataGenerator(rescale=1./255) 
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up batch size and number of epochs for training
batch_size = 32
train_generator = train_datagen.flow_from_directory(
train_path,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
val_path,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
test_path,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='categorical')

checkpoint = ModelCheckpoint("F:/Project 2/spectograms/Train test/weights19t5.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


history = model.fit_generator(
train_generator,
steps_per_epoch=train_generator.n // batch_size,
epochs=3,
validation_data=val_generator,
validation_steps=val_generator.n // batch_size,
callbacks=[checkpoint])

model.load_weights("F:/Project 2/spectograms/Train test/weights19t5.h5")


test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n // batch_size)
print('Test accuracy:', test_acc)

test_generator.reset()
predictions = model.predict_generator(test_generator, steps=test_generator.n // batch_size, verbose=1)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.show()

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

#========================================================================================
# Load the image
img = image.load_img('F:/Project 2/spectograms/Train test/gum_00856_00213787811.png', target_size=(img_width, img_height))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Normalize the image data
img_array = img_array / 255.0

# Add a batch dimension to the image data
img_array = np.expand_dims(img_array, axis=0)

# Use the model to make a prediction
prediction = model.predict(img_array)

# Get the predicted class
predicted_class = np.argmax(prediction)

# Print the predicted class
print(f"Predicted class: {predicted_class}")



# Plot validation loss trend
plt.plot(history.history['val_loss'])
plt.title('Validation Loss Trend')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.show()

# plotting val_accu
plt.plot(history.history['val_accuracy'])
plt.title('Validation Accuracy Trend')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.show()






preds = model.predict(test_generator, verbose=1) 
# find the index of the class with the highest probability for each prediction.

pred_labels = np.argmax(preds, axis=1)
# argmax function to find the index of the class with the highest probability for each prediction. 

true_labels = test_generator.classes
# Get the true labels from the test generator, The classes attribute of the generator returns an array of integers,


class_names = list(test_generator.class_indices.keys())
# Get the class names from the test generator
# class_indices attribute of the generator returns a dictionary that maps each class name to its corresponding integer label



# Compute the confusion matrix and classification report
confusion_mat = confusion_matrix(true_labels, pred_labels)
classification_rep = classification_report(true_labels, pred_labels, target_names=class_names)

# Print the classification report
print(classification_rep)

# Create a heatmap to visualize the confusion matrix
sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)

# Add axis labels and a title
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

# Display the plot
plt.show()


