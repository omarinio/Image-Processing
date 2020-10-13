#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() { 

    // Read image from file
    Mat image = imread("mandrill.jpg", CV_LOAD_IMAGE_UNCHANGED);

    // when returning lowpass remember to divide by 9,  img.at<uchar>(y-1, x-1) = sum/9
    Mat lowpass = Mat::ones(3, 3, CV_32F);
    
    Mat sharpen(3, 3, CV_32F, -1);
    sharpen.at<float>(1,1) = 9;

    Mat img = Mat::zeros(Size(image.cols-2,image.rows-2), CV_8UC1);

    // Threshold by looping through all pixels
    for(int y=1; y<image.rows-1; y++) {
        for(int x=1; x<image.cols-1; x++) {
            float sum = 0;
            for (int i = -1; i < 2; i++) {
                for (int j = -1; j < 2; j++) {
                    sum += image.at<uchar>(y - i, x - j) * sharpen.at<float>(i+1, j+1);
                }
            }

            img.at<uchar>(y-1,x-1) = sum;
        }
    }

    namedWindow("Display window", CV_WINDOW_AUTOSIZE);

    imshow("Display window", img);

    std::cout << "kernel: "<< std::endl << sharpen << std::endl<< std::endl;
    //Save thresholded image
    imwrite("sharpened.jpg", img);

    waitKey(0);

    img.release();

  return 0;
}