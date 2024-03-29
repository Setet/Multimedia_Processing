import cv2
import numpy as np


# Определите функцию, первый параметр - это коэффициент масштабирования,
# а второй параметр - это кортеж или список изображений, которые будут отображаться
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

    summ_1 = 0
    for i in range(n):
        for j in range(n):
            g_m[i][j] = (1 / (2 * np.pi * sigma ** 2) * np.exp(
                -((i - pad) ** 2 + (j - pad) ** 2) / (2 * sigma ** 2)))
            summ_1 += g_m[i][j]

    summ_2 = 0
    for i in range(n):
        for j in range(n):
            g_m[i][j] = g_m[i][j] / summ_1
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
            print(newGray[i][j])

    return newGray


#   Реализовать фильтр Гаусса средствами языка python.
def task_1(size, sigma):
    image = cv2.imread(r'F:\Material_Multimedia_Processing\456.jpg')

    stackedimageb = ManyImgs(0.3, ([image, GF(image, size, sigma)]))

    cv2.namedWindow("Laba_2")
    cv2.imshow("Laba_2", stackedimageb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#   Применить данный фильтр для двух разных значений
#   среднего квадратичного отклонения и двух разных размерностей матрицы
#   свертки, сравнить результаты для ОДНОГО изображения.
def task_2(size, sigma):
    image = cv2.imread(r'F:\Material_Multimedia_Processing\456.jpg')
    image_blur_3 = GF(image, size, sigma)
    image_blur_5 = GF(image, size + 2, sigma + 0.1)
    image_blur_11 = GF(image, size + 4, sigma + 0.2)

    stackedimageb = ManyImgs(0.3, ([image, image_blur_3], [image_blur_5, image_blur_11]))

    cv2.namedWindow("Laba_2")
    cv2.imshow("Laba_2", stackedimageb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#   Реализовать размытие Гаусса встроенным методом
#   библиотеки OpenCV, сравнить результаты с Вашей реализацией.
def task_3(size, sigma):
    image = cv2.imread(r'F:\Material_Multimedia_Processing\123.jfif')
    img_blur_7 = cv2.GaussianBlur(image, (size, size), sigma)
    image_blur_7 = GF(image, size, sigma)

    Blankimg = np.zeros((200, 200), np.uint8)  # Размер может быть принудительно преобразован любой функцией
    stackedimageb = ManyImgs(0.3, ([image, Blankimg], [img_blur_7, image_blur_7]))

    cv2.namedWindow("stackedimageb")
    cv2.imshow("stackedimageb", stackedimageb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#   Реализовать размытие Гаусса средствами
#   любого другого языка программирования.
#   Ещё делается(

def main():
    size = 7
    sigma = 0.8
    task_3(size, sigma)


if __name__ == '__main__':
    main()
