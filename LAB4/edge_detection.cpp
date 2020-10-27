// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

void Sobel(
	cv::Mat &input, 
	int size,
	cv::Mat &magOutput,
    cv::Mat &xOutput,
    cv::Mat &yOutput,
    cv::Mat &dirOutput);

void Normalisation(
    cv::Mat &input, 
    cv::Mat &output);

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

    Mat xSobel; 
    Mat ySobel;
    Mat magSobel; 
    Mat dirSobel;          
    Sobel(gray_image, 3, magSobel, xSobel, ySobel, dirSobel);
    imwrite( "SobelXDirection2.jpg", xSobel );
    imwrite( "SobelYDirection2.jpg", ySobel );
    imwrite( "SobelMag2.jpg", magSobel );
    imwrite( "SobelDir2.jpg", dirSobel );

    Mat xNormalise;
    Mat yNormalise;
    Mat magNormalise;
    Mat dirNormalise;
    Normalisation(xSobel, xNormalise);
    Normalisation(ySobel, yNormalise);
    Normalisation(magSobel, magNormalise);
    Normalisation(dirSobel, dirNormalise);
    imwrite( "NormaliseX2.jpg", xNormalise );
    imwrite( "NormaliseY2.jpg", yNormalise );
    imwrite( "NormaliseMag2.jpg", magNormalise );
    imwrite( "NormaliseDir2.jpg", dirNormalise );

    return 0;
}

void Sobel(cv::Mat &input, int size, cv::Mat &magOutput, cv::Mat &xOutput, cv::Mat &yOutput, cv::Mat &dirOutput) {
	// intialise the output using the input
	magOutput.create(input.size(), input.type());
    xOutput.create(input.size(), input.type());
    yOutput.create(input.size(), input.type());
    dirOutput.create(input.size(), input.type());

    double kernelX[3][3] = {{1, 0, -1},
                            {2, 0, -2},
                            {1, 0, -1}};

    double kernelY[3][3] = {{1, 2, 1},
                            {0, 0, 0},
                            {-1, -2, -1}};

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( size - 1 ) / 2;
	int kernelRadiusY = ( size - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			double sumX = 0.0;
            double sumY = 0.0;

			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalvalX = kernelX[kernelx][kernely];
                    double kernalvalY = kernelY[kernelx][kernely];

					// do the multiplication
					sumX += imageval * kernalvalX;		
                    sumY += imageval * kernalvalY;					
				}
			}
			// set the output value as the sum of the convolution
			xOutput.at<uchar>(i, j) = (uchar) sumX;
            yOutput.at<uchar>(i, j) = (uchar) sumY;

            magOutput.at<uchar>(i, j) = (uchar) sqrt((sumX * sumX) + (sumY * sumY));

            dirOutput.at<uchar>(i, j) = (uchar) atan2(sumY, sumX);
		}
	}
}

void Normalisation(cv::Mat &input, cv::Mat &output) {
    output.create(input.size(), input.type());
    double minVal; 
    double maxVal; 
    minMaxLoc( input, &minVal, &maxVal );
    std::cout << minVal << std::endl;
    std::cout << maxVal << std::endl;

    double newMin = 0;
    double newMax = 255;

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            int imageval = ( int ) input.at<uchar>( i, j );
            double newIm = (imageval - minVal) * ((newMax - newMin)/(maxVal - minVal)) + newMin;
            output.at<uchar>(i, j) = (uchar) newIm;
        }
    }
}
