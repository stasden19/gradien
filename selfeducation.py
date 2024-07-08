import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)


model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 1
num_epochs = 1
num_batches = len(x_train)


plt.ion()
fig = plt.figure()
for batch in range(num_batches):
    fig.clear()
    x_batch = x_train[batch:batch + batch_size]
    y_batch = y_train_cat[batch:batch + batch_size]

    plt.imshow(x_batch[0])
    fig.canvas.draw()
    fig.canvas.flush_events()
    x_batch = np.expand_dims(x_batch, axis=4)
    print(f'Я думаю что это: {np.argmax(model.predict_on_batch(x_batch))}\nЕсли я права, нажми Enter или введи правильную цифру\nДля сохранения введи "save"')
    number = input('Ввод: ')

    if number == 'save':
        model.save(f'model{batch}.h5')
    elif number:
        y_batch = np.expand_dims(keras.utils.to_categorical(int(number), 10), axis=0)
    loss, accuracy = model.train_on_batch(x_batch, y_batch)
    print(f'Batch {batch + 1}/{num_batches}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

plt.ioff()
plt.show()
model.evaluate(x_test, y_test_cat)
