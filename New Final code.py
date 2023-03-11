# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Define the VGG16 model
model = keras.applications.vgg16.VGG16(
    include_top=False, input_shape=(256, 256, 3), weights='imagenet'
)

# Freeze the pre-trained layers
for layer in model.layers:
    layer.trainable = False

# Add new layers to the model
x = layers.Flatten()(model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(3, activation='softmax')(x)

# Create the new model
new_model = keras.models.Model(inputs=model.input, outputs=x)

# Compile the model
new_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the data
train_datagen = ImageDataGenerator(rescale=1./255) #shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen= ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('F:/Project 2/spectograms/Train test/Train/', target_size=(256, 256), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('F:/Project 2/spectograms/Train test/Test/', target_size=(256, 256), batch_size=32, class_mode='categorical')
val_set = val_datagen.flow_from_directory('F:/Project 2/spectograms/Train test/Val/', target_size=(256, 256), batch_size=32, class_mode='categorical')


# Train the model
new_model.fit(train_set, epochs=3, validation_data=val_set)

# Evaluate the model
loss, accuracy = new_model.evaluate(test_set)
print('Test accuracy:', accuracy)

