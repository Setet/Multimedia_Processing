import os
import random
import numpy as np
import matplotlib.pyplot as plt

# библиотека базы выборок Mnist
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных(1 или 0)
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),  # Входные данных 28*28 = 784
    Dense(128, activation='relu'),  # У нейронной сети 128 нейронов входные
    Dense(10, activation='softmax')  # и 10 выходных
])

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])  # Метрика уменьшения ошибки

model.fit(x_train, y_train_cat, batch_size=64, epochs=1, validation_split=0.5)

print("Критерий проверки качества\n")
# model.evaluate(x_test, y_test_cat)

n = random.randint(0, 49)
print("Элемент массива = " + str(n + 1))
x = np.expand_dims(x_test[n], axis=0)
print("Цифра :" + str(np.argmax(model.predict(x))))

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()
