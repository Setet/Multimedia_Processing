import cv2


def main():
    border = 10
    video = cv2.VideoCapture("F:/Material_Multimedia_Processing/LR_4.mov")
    ret, frame = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("out.mp4", fourcc, 25, (w, h))
    gray1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (11, 11), 0)

    while True:

        ret, frame2 = video.read()
        if ret == False:
            break
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (11, 11), 0)

        frameabsdiff = cv2.absdiff(gray2, gray1)
        _, thresh = cv2.threshold(frameabsdiff, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:

            if cv2.contourArea(i) > border:
                video_writer.write(frame)
                break

        cv2.imshow('img', frame)
        frame = frame2
        gray1 = gray2
        if cv2.waitKey(25) & 0xFF == 27:
            break
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
