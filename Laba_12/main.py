import os
import random

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # x_train - Изоброжение цифр y_train - Вектор

# стандартизация входных данных
x_train = x_train / 255  # Значения будут хранить 1 или 0
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)  # Вектор 10 который будет хранить категорию
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Показывает первые 25 изображения
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),  # Входые данных 784
    Dense(128, activation='relu'),  # У нейронки 128 нейронов входные
    Dense(10, activation='softmax')  # и 10 выходных
])

print(model.summary())  # вывод структуры НС в консоль

model.compile(optimizer='adam',  # Оптимизация по методу адам
              loss='categorical_crossentropy',  # Функция потерь по функции котегориальная кросетропия
              metrics=['accuracy'])  # Метрика уменьшения ошибки

model.fit(x_train, y_train_cat, batch_size=32, epochs=3, validation_split=0.5)  # Эпоки, и 20% данных

model.evaluate(x_test, y_test_cat)  # Критерий проверки качества

n = random.randint(0, 24)
print("Элемент массива = " + str(n+1))
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print("Цифра :" + str(np.argmax(res)))

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()
