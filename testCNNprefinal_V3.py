import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras import utils
from keras.preprocessing.image import ImageDataGenerator

utils.set_random_seed(0)

datagen = ImageDataGenerator(rescale=1/255, validation_split=0.25)
batch_size = 64
train_generator = datagen.flow_from_directory(
    'git_conn/train_m3',
    subset='training',
    target_size=(240, 80),
    batch_size=batch_size,
    class_mode='categorical'
    )

val_generator = datagen.flow_from_directory(
    'git_conn/train_m3',
    subset='validation',
    target_size=(240, 80),
    batch_size=batch_size,
    class_mode='categorical'
    )

test_generator = datagen.flow_from_directory(
    'git_conn/test_m3',
    target_size=(240, 80),
    batch_size=batch_size,
    class_mode='categorical'
    )

# モデル定義
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(240, 80, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 4)))
model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics='accuracy')

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100
)


result = pd.DataFrame(history.history)

result[['loss','val_loss']].plot(ylim=[0, 2])

result[['accuracy', 'val_accuracy']].plot(ylim=[0, 1])

plt.show()

