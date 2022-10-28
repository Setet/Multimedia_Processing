#include <opencv2/opencv.hpp>
#include <iostream>
#include "vector"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    Mat frame,framehsv;
    Vec3b vec;
    VideoCapture cap(0);

    cap >> frame;
    cvtColor(frame,framehsv,COLOR_RGB2HSV);

    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

    int centrW = w/2;
    int centrH = h/2;

    Point pt1(centrW+100,centrH);
    Point pt2(centrW-100,centrH+3);
    Point pt3(centrW,centrH+100);
    Point pt4(centrW+3,centrH-100);

    double fps = cap.get(CAP_PROP_FPS);

    if (!cap.isOpened())
    {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    for (;;)
    {
        if (!cap.read(frame))
        {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
    vec = framehsv.at<Vec3b>(centrH,centrW);
    Vec3b vec2;
    int sec = 180/3;
    int sec2 = sec/2;
    if((vec[0]>sec2) && (vec[0]<(sec2+sec)))
    {
        vec2[0] =0;
        vec2[1] = 255;
        vec2[2] = 0;
    }else if((vec[0]>sec2+sec) && (vec[0]<255-sec2)){
        vec2[0] =255;
        vec2[1] = 0;
        vec2[2] = 0;
    }else{
        vec2[0] =0;
        vec2[1] = 0;
        vec2[2] = 255;
    }

    cvtColor(frame,framehsv,COLOR_BGR2HSV);
    rectangle(frame,pt1,pt2,vec2,FILLED);
    rectangle(frame,pt3,pt4,vec2,FILLED);
    imshow("Live", frame);
    if (waitKey(1000/fps) >= 27)
    {
        break;
    } 
    }
return 0;
}