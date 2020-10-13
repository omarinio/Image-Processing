/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - thr.cpp
// TOPIC: RGB explicit thresholding
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() { 

    // Read image from file
    Mat image = imread("mandrill1.jpg", 1);

    Mat image2 = imread("mandrill1.jpg", 1);

    // Threshold by looping through all pixels
    for(int y=0; y<image.rows; y++) {
        for(int x=0; x<image.cols; x++) {
            uchar pixelRed = image.at<Vec3b>(y,x)[2];

            image2.at<Vec3b>((y+32) % image2.rows,(x+32) % image2.cols)[2] = pixelRed;
        }
    }

    namedWindow("Display window", CV_WINDOW_AUTOSIZE);

    imshow("Display window", image2);

    //Save thresholded image
    imwrite("fixed1.jpg", image2);

    waitKey(0);

    image.release();

  return 0;
}