import math

import cv2
import numpy as np


def ManyImgs(scale, imgarray):
    rows = len(imgarray)  # Длина кортежа или списка
    cols = len(imgarray[
                   0])  # Если imgarray - это список, вернуть количество каналов первого изображения в списке, если это кортеж, вернуть длину первого списка, содержащегося в кортеже
    # print("rows=", rows, "cols=", cols)

    # Определить, является ли тип imgarray [0] списком
    # Список, указывающий, что imgarray является кортежем и должен отображаться вертикально
    rowsAvailable = isinstance(imgarray[0], list)

    # Ширина и высота первой картинки
    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]
    # print("width=", width, "height=", height)

    # Если входящий кортеж
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # Обойти кортеж, если это первое изображение, не преобразовывать
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (0, 0), None, scale, scale)
                # Преобразуйте другие матрицы к тому же размеру, что и первое изображение, и коэффициент масштабирования будет масштабироваться
                else:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]),
                                                None, scale, scale)
                # Если изображение в оттенках серого, преобразовать его в цветное отображение
                if len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(imgarray[x][y], cv2.COLOR_GRAY2BGR)

        # Создайте пустой холст того же размера, что и первое изображение
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [
                  imgBlank] * rows  # Тот же размер, что и первое изображение, и столько же горизонтальных пустых изображений, сколько кортеж содержит список
        for x in range(0, rows):
            # Расположить x-й список в кортеже по горизонтали
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)  # Объединить разные списки по вертикали
    # Если входящий - это список
    else:
        # Операция трансформации, как и раньше
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        # Расположить список по горизонтали
        hor = np.hstack(imgarray)
        ver = hor
    return ver


def GF(img, size, sigma):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    newGray = img

    newGray = cv2.cvtColor(newGray, cv2.COLOR_BGR2GRAY)

    h, w = newGray.shape[:2]

    n = size

    pad = n // 2
    g_m = [[0] * n for i in range(n)]

    summ = 0
    for i in range(n):
        for j in range(n):
            g_m[i][j] = (1 / (2 * np.pi * sigma ** 2) * np.exp(
                -((i - pad) ** 2 + (j - pad) ** 2) / (2 * sigma ** 2)))
            summ += g_m[i][j]

    summ_2 = 0
    for i in range(n):
        for j in range(n):
            g_m[i][j] = g_m[i][j] / summ
            summ_2 += g_m[i][j]

    finishh = h - pad
    finishw = w - pad

    for i in range(pad, finishh):
        for j in range(pad, finishw):
            new_value = 0
            for k in range(n):
                for l in range(n):
                    new_value = new_value + g_m[k][l] * gray[i - pad + k][j - pad + l]
            newGray[i][j] = new_value

    return newGray


def tang(x, y):
    tg = y / x

    a = 0

    if (x > 0 and y < 0 and tg < -2.414) or (x < 0 and y < 0 and tg > 2.414): a = 0
    if x > 0 and y < 0 and tg < -0.414: a = 1
    if (x > 0 and y < 0 and tg < -0.414) or (x > 0 and y > 0 and tg < 0.414): a = 2
    if x > 0 and y > 0 and tg < 2.414: a = 3
    if (x > 0 and y > 0 and tg > 2.414) or (x < 0 and y > 0 and tg > -2.414): a = 4
    if x < 0 and y > 0 and tg < -0.414: a = 5
    if (x < 0 and y > 0 and tg < -0.414) or (x < 0 and y < 0 and tg > 0.414): a = 6
    if x < 0 and y < 0 and tg < 2.414: a = 7

    return a


def is_border(matrix, max):
    for i in range(2):
        for j in range(2):
            if matrix[i][j] > max:
                return True
    return False


def main():
    size = 5
    sigma = 0.5
    high = 80
    low = 40

    img1 = cv2.imread(r'F:\Material_Multimedia_Processing\LR3.jpg')
    image_GF = GF(img1, size, sigma)

    h, w = image_GF.shape[:2]

    image_GF1 = image_GF.copy()
    image_GF2 = [[0] * w for i in range(h)]

    csize = 3
    pad = csize // 2

    finishh = h - pad
    finishw = w - pad

    g_m_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    g_m_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    for i in range(pad, finishh):
        for j in range(pad, finishw):
            valueX = 0
            valueY = 0
            for k in range(csize):
                for l in range(csize):
                    valueX = valueX + g_m_x[k][l] * image_GF[i - pad + k][j - pad + l]
                    valueY = valueY + g_m_y[k][l] * image_GF[i - pad + k][j - pad + l]

            image_GF1[i][j] = math.sqrt(valueX ** 2 + valueY ** 2)
            image_GF2[i][j] = tang(valueX, valueY)

    for i in range(pad, finishh):
        for j in range(pad, finishw):
            if image_GF2[i][j] == 0 | image_GF2[i][j] == 4:
                if image_GF1[i][j + 1] < image_GF1[i][j] & image_GF1[i][j] > image_GF1[i][j - 1]:
                    image_GF1[i][j] = image_GF1[i][j]
                else:
                    image_GF1[i][j] = 0
            if image_GF2[i][j] == 1 | image_GF2[i][j] == 5:
                if image_GF1[i + 1][j + 1] < image_GF1[i][j] & image_GF1[i][j] > image_GF1[i - 1][j - 1]:
                    image_GF1[i][j] = image_GF1[i][j]
                else:
                    image_GF1[i][j] = 0
            if image_GF2[i][j] == 2 | image_GF2[i][j] == 6:
                if image_GF1[i + 1][j] < image_GF1[i][j] & image_GF1[i][j] > image_GF1[i - 1][j]:
                    image_GF1[i][j] = image_GF1[i][j]
                else:
                    image_GF1[i][j] = 0
            if image_GF2[i][j] == 3 | image_GF2[i][j] == 7:
                if image_GF1[i + 1][j - 1] < image_GF1[i][j] & image_GF1[i][j] > image_GF1[i - 1][j + 1]:
                    image_GF1[i][j] = image_GF1[i][j]
                else:
                    image_GF1[i][j] = 0

    for i in range(pad, finishh):
        for j in range(pad, finishw):
            if image_GF1[i][j] > high:
                image_GF1[i][j] = 255
            elif image_GF1[i][j] > low:
                if is_border(image_GF1[i - 1:i + 2, j - 1:j + 2], high):
                    image_GF1[i][j] = 255
                else:
                    image_GF1[i][j] = 0
            else:
                image_GF1[i][j] = 0

    cv2.namedWindow("Lab_3", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Lab_3", image_GF1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
