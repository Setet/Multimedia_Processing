#   Установить библиотеку OpenCV.
import cv2


#   Вывести на экран изображение. Протестировать три
#   возможных расширения, три различных флага для создания окна и три
#   различных флага для чтения изображения.
def task_2():
    img1 = cv2.imread(r'F:\Material_Multimedia_Processing\123.jpg')
    cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)
    cv2.imshow('Display window', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#   Отобразить видео в окне. Рассмотреть методы класса
#   VideoCapture и попробовать отображать видео в разных форматах, в частности
#   размеры и цветовая гамма.
def task_3():
    cap = cv2.VideoCapture(r'F:\Material_Multimedia_Processing\Rick_Astley_Never_Gonna_Give_You_Up.mpg', cv2.CAP_ANY)
    while True:
        ret, frame = cap.read()
        if not (ret):
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


#   Записать видео из файла в другой файл.
def task_4():
    video = cv2.VideoCapture(r'F:\Material_Multimedia_Processing\Rick_Astley_Never_Gonna_Give_You_Up.mpg')
    ok, img = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video_writer = cv2.VideoWriter(r'F:\Material work\Rick.mpg', fourcc, 25, (w, h))

    while (True):
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()


#   Отобразить информацию с вебкамеры,
#   записать видео, продемонстрировать видео на следующем занятии.
def task_5():
    video = cv2.VideoCapture(0)
    ok, img = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video_writer = cv2.VideoWriter(r'F:\Material work\Rick.mpg', fourcc, 25, (w, h))

    while (True):
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()


#   Подключите телефон, подключитесь к его
#   камере, протранслируйте запись с камеры. Продемонстрировать процесс на
#   ноутбуке преподавателя и своем телефоне.
def task_6():
    video = cv2.VideoCapture(1)
    ok, img = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video_writer = cv2.VideoWriter(r'F:\Material work\Rick.mpg', fourcc, 25, (w, h))

    while (True):
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()


def main():
    task_6()


if __name__ == '__main__':
    main()
