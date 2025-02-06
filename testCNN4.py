# trainデータを新調（train_m2)にしたモデル
# train_m2 は make_traindata2.ipynbにて作成
# ImageDataGenerator の　generator.flowを使って、train　データを増やす
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras import utils
from keras.preprocessing.image import load_img,img_to_array
utils.set_random_seed(0)

# trainデータをnp.arrayに変換
train_m2_list = glob.glob("git_conn/train_m2" + "/*." + "png")
train_m2_list.sort()
x_train_m2_list = []
for img in train_m2_list:
    tmp_img = load_img(img)
    tmp_img_array = img_to_array(tmp_img)
    x_train_m2_list.append(tmp_img_array)
x_train_m2 = np.array(x_train_m2_list)

# testデータをnp.arrayに変換
test_m2_list = glob.glob("git_conn/test_m2" + "/*." + "png")
test_m2_list.sort()
x_test_m2_list = []
for img in test_m2_list:
    tmp_img = load_img(img)
    tmp_img_array = img_to_array(tmp_img)
    x_test_m2_list.append(tmp_img_array)
x_test_m2 = np.array(x_test_m2_list)

# 正解ラベル
y_train_m2 = pd.read_csv("git_conn/y_train_m2.csv",header=None)
y_test_m2 = pd.read_csv("git_conn/y_test_m2.csv",header=None)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        #回転
    rotation_range = 0,
    #左右反転
    horizontal_flip = False,
    #上下平行移動
    height_shift_range = 0,
    #左右平行移動
    width_shift_range = 0.6,
    #ランダムにズーム
    zoom_range = 0,
    #チャンネルシフト
    channel_shift_range = 0,
    #スケーリング
    rescale = 1./255
    #validation split
    #validation_split=0.25
    )

testgen = ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow(
        x_train_m2,
        y_train_m2,
        batch_size=16,
        seed=0
        #subset='training'
        )

# val_generator = datagen.flow(
#         'git_conn/train_m2',
#         y_train_m2,
#         batch_size=16,
#         seed=0,
#         subset='validation',
#         )

test_generator = testgen.flow(
        x_test_m2,
        y_test_m2,
        batch_size=16
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
    validation_data=test_generator,
    epochs=60
)

import pandas as pd
result = pd.DataFrame(history.history)

model.evaluate(test_generator)


