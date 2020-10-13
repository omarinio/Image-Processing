#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() {
    Mat image;

    image = imread("mandrill.jpg", CV_LOAD_IMAGE_UNCHANGED);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            uchar pixel = image.at<uchar>(y,x);
            if (pixel < 128) {
                image.at<uchar>(y,x) = 0;
            } else {
                image.at<uchar>(y,x) = 255;
            }
        }
    }

    namedWindow("Display window", CV_WINDOW_AUTOSIZE);

    imshow("Display window", image);

    imwrite("monke.jpg", image);

    waitKey(0);

    image.release();

    return 0;
}