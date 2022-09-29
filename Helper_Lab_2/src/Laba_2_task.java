import org.opencv.core.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;

public class Laba_2_task {

    public static void main(String[] args) {
        try{
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            Mat src=Imgcodecs.imread("data/lenna.jpg");
            // Считываем изображение в матрицу
            if(src.empty()){
                throw new Exception("no file");
            }

            Mat dst = src.clone();
            // Копируем матрицу в dst

            Imgproc.GaussianBlur(src,dst,new Size(13,13),10,10);
            // Обработка размытия изображения 11
            Imgcodecs.imwrite("./data/gaussianblur1.jpg", dst);

            Imgproc.GaussianBlur(src,dst,new Size(31,5),80,3);
            // Обработка размытия изображения 33
            Imgcodecs.imwrite("./data/gaussianblur2.jpg", dst);
        }catch(Exception e){
            System.out.println ("Исключение:" + e);
        }

    }

}