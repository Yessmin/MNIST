# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:56:38 2020

@author: gourya
"""


import tensorflow as tf # deep learning library. Tensors are a multi-dimensional arrays
import tensorflow.keras as keras # a library for developing and evaluating deep learning models
import numpy as np
import matplotlib.pyplot as plt # used to picture the digits 

mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten numbers and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # drain images to x_train/x_test and labels to y_train/y_test

plt.imshow(x_train[0],cmap=plt.cm.binary) # we add cmap=plt.cm.binary to ommit colors(jupyter)
plt.show()

# normalizing the data generally speeds up learning and leads to faster convergence
x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
# now each pixel is less than 1 which explains the fading color

# time to build the model
model = tf.keras.models.Sequential()  
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784 , we can also use reshape
# hidden layers :  
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 128 units, relu activation
# output layer :
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # 10 units for 10 classes(from 0 to 9). Softmax for probability distribution

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

model.fit(x_train, y_train, epochs=5)  # train the model

val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(" YOUR MODEL'S LOSS IS :",val_loss)  # model's loss (error)
print(" YOUR MODEL'S ACCURACY IS :",val_acc)  # model's accuracy

model.save('num_reader.model') # save the model to use it later
new_model = tf.keras.models.load_model('num_reader.model')

new_predictions = new_model.predict(x_test)
print(new_predictions) # this returns one hot arrays which represents the probability distribution that's why we select the index of the maximum value
print(np.argmax(new_predictions[3]))
plt.imshow(x_test[3],cmap=plt.cm.binary)
plt.show()