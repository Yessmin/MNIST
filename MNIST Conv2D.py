# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:24:21 2020

@author: goury
"""


import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 12

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print("Training samples: {}".format(x_train.shape[0]))
print("Test samples: {}".format(x_test.shape[0]))
# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


x_val = x_train[:15000]
x_train=x_train[45000:]
y_val = y_train[:15000]
y_train=y_train[45000:]

Final_model=model.fit(x_train, y_train, epochs=12, batch_size=batch_size, validation_data=(x_val,y_val)) 
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

Final_model.history
result = model.evaluate(x_test, y_test, verbose=0)
losstest=result[0]
acctest=result[1] 
losstrain=Final_model.history['loss']
lossval=Final_model.history['val_loss']
acctrain=Final_model.history['accuracy']
accval=Final_model.history['val_accuracy']
 
# Scores

print("Test loss :",losstest,", +/-", losstest)
print("Test accuracy :",acctest,", +/-", acctest)


# Plots

plt.subplot(121)
plt.plot(losstrain, label='Training loss')
plt.plot(lossval, label='Validation loss')
plt.title('Loss variation')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.subplot(122)
plt.plot(acctrain, label='Training accuration')
plt.plot(accval, label='Validation accuration')

plt.title('Accuracy variation')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.show()