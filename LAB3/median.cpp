// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

void MedianFilter(
	cv::Mat &input, 
	int size,
	cv::Mat &blurredOutput);

int main( int argc, char** argv ) {

    // LOADING THE IMAGE
    char* imageName = argv[1];

    Mat image;
    image = imread( imageName, 1 );

    if( argc != 2 || !image.data ) {
    printf( " No image data \n " );
    return -1;
    }

    // CONVERT COLOUR, BLUR AND SAVE
    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );

    Mat carBlurred;
    MedianFilter(gray_image,5,carBlurred);

    imwrite( "car2Median.jpg", carBlurred );

    return 0;
}

void MedianFilter(cv::Mat &input, int size, cv::Mat &medianOutput) {
	// intialise the output using the input
	medianOutput.create(input.size(), input.type());

	int radius = (size-1)/2;
    cv::Mat paddedInput;
    cv::copyMakeBorder( input, paddedInput, radius, radius, radius, radius, cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			vector<double> vect;
			for( int m = -radius; m <= radius; m++ ) {
				for( int n = -radius; n <= radius; n++ ) {
                    int imagex = i + m + radius;
                    int imagey = j + n + radius;
                    vect.push_back(( double ) paddedInput.at<uchar>(imagex, imagey));
				}
			}
			std::sort(vect.begin(), vect.end());

            size_t size = vect.size();

			medianOutput.at<uchar>(i, j) = (uchar) vect[(size)/2];
		}
	}
}
