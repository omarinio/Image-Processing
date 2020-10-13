#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() {
    Mat image;

    image = imread("mandrillRGB.jpg", CV_LOAD_IMAGE_UNCHANGED);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // [0] = blue, [1] = green [2] = red
            uchar blue = image.at<Vec3b>(y,x)[0];
            uchar green = image.at<Vec3b>(y,x)[1];
            uchar red = image.at<Vec3b>(y,x)[2];

            if (blue > 200) {
                image.at<Vec3b>(y,x)[0] = 255;
                image.at<Vec3b>(y,x)[1] = 255;
                image.at<Vec3b>(y,x)[2] = 255;
            } else {
                image.at<Vec3b>(y,x)[0] = 0;
                image.at<Vec3b>(y,x)[1] = 0;
                image.at<Vec3b>(y,x)[2] = 0;
            }


        }
    }
    namedWindow("Display window", CV_WINDOW_AUTOSIZE);

    imshow("Display window", image);

    imwrite("monkeRGB.jpg", image);

    waitKey(0);

    image.release();

    return 0;

}