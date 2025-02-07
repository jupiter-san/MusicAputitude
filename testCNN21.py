import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import utils
utils.set_random_seed(0)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1/255,validation_split=0.25)

train_generator = datagen.flow_from_directory(
        'git_conn/train',
        subset='training',
        target_size=(80, 240),
        batch_size=32,
        class_mode='categorical'
        )

val_generator = datagen.flow_from_directory(
        'git_conn/train',
        subset='validation',
        target_size=(80, 240),
        batch_size=32,
        class_mode='categorical'
        )

test_generator = datagen.flow_from_directory(
        'git_conn/test',
        target_size=(80, 240),
        batch_size=32,
        class_mode='categorical'
        )


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
# モデル定義
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(80, 240, 3)))
model.add(MaxPooling2D(pool_size=(5, 3)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 3)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100
)

import pandas as pd
result = pd.DataFrame(history.history)

result[['loss','val_loss']].plot(ylim=[0, 2])

result[['accuracy', 'val_accuracy']].plot(ylim=[0, 1])

plt.show()

