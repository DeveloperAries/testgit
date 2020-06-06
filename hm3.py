import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D,concatenate,Input,Activation
from tensorflow.keras.utils import plot_model
import time
import os
num_classes=10
mnist = tf.keras.datasets.mnist

#1. prepare datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.one_hot(y_train, num_classes)
y_test = tf.one_hot(y_test, num_classes)
x_train=x_train.reshape((60000,-1))
x_train_1 = x_train[:,:392]
x_train_2 = x_train[:,392:]
x_test=x_test.reshape((10000,-1))
x_test_1 = x_test[:,:392]
x_test_2 = x_test[:,392:]
#2. net_build
a=tf.keras.Input(shape=(392,))

O=Dense(units= 10,input_dim = 392, activation = None, use_bias = False,name = 'D1')(a)
shared_base=Model(inputs = a,outputs = O ,name = 'seq1')

x1 = tf.keras.Input(shape=(392,),name='Input1')
x2 = tf.keras.Input(shape=(392,),name='Input2')
s1=shared_base(x1)
s2=shared_base(x2)

b = K.zeros(shape=(10))
x = s1+s2+b
s = tf.keras.layers.Activation(activation='softmax',name='softmax')(x)

siamese_net = Model(inputs = [x1,x2], outputs = s,name = 'siamese_net')

plot_model(siamese_net, to_file='./siamese_net.png', show_shapes=True,expand_nested=True)

#3. train and test
'''
#方法一：compile+fit
siamese_net.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001),
                    metrics=['accuracy'])
siamese_net.fit([x_train_1, x_train_2], y_train, epochs=10, batch_size=64,
                validation_data=([x_test_1, x_test_2], y_test))
'''

#方法二：自定义循环
train_ds = tf.data.Dataset.from_tensor_slices(((x_train_1, x_train_2), y_train)).shuffle(10000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices(((x_test_1, x_test_2), y_test)).shuffle(10000).batch(64)
optimizer = tf.keras.optimizers.Adam(0.001)
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
loss_object = tf.keras.losses.categorical_crossentropy


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = siamese_net(images)
        print(predictions)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, siamese_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, siamese_net.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = siamese_net(images)
    loss = loss_object(labels, predictions)
    test_loss(loss)
    test_accuracy(labels, predictions)


EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)
    for images, labels in test_ds:
        test_step(images, labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

#4. draw weights of 10 classes
train_weights=siamese_net.get_layer('seq1').get_layer('D1').kernel.numpy()


num = np.arange(0, 392, 1, dtype="float")
num = num.reshape((14, 28))
plt.figure(num='Weights', figsize=(10, 10))  # 创建一个名为Weights的窗口,并设置大小
for i in range(10):  # W.shape[1]
    num = train_weights[:, i: i+1].reshape((14, -1))
    plt.subplot(2, 5, i + 1)
    num = num * 255.
    plt.imshow(num, cmap=plt.get_cmap('hot'))
    plt.title('weight %d image.' % (i + 1))  # 第i + 1幅图片
plt.show()
print(np.min(num))
print(np.max(num))








