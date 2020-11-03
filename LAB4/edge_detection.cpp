// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

#define MAXRADIUS 100
#define MINRADIUS 20

void Sobel(
	cv::Mat &input, 
	int size,
	cv::Mat &magOutput,
    cv::Mat &xOutput,
    cv::Mat &yOutput,
    cv::Mat &dirOutput);

void GaussianBlur(
	cv::Mat &input, 
	int size,
	cv::Mat &blurredOutput);

Mat threshold(
    cv::Mat &input);

void hough(
    Mat &input,
    Mat &gradient, 
    Mat &direction, 
    int ***houghSpace,
    int radius);

int ***malloc3dArray(int dim1, int dim2, int dim3) {
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	    for (j = 0; j < dim2; j++) {
  	        array[i][j] = (int *) malloc(dim3 * sizeof(int));
	    }

    }
    return array;
}


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

    Mat blurred;
    GaussianBlur(gray_image,8,blurred);

    Mat xSobel(image.rows,image.cols, CV_32FC1);
    Mat ySobel(image.rows,image.cols, CV_32FC1);
    Mat magSobel(image.rows,image.cols, CV_32FC1);
    Mat dirSobel(image.rows,image.cols, CV_32FC1);         
    Sobel(blurred, 3, magSobel, xSobel, ySobel, dirSobel);
    imwrite( "SobelXDirection.jpg", xSobel );
    imwrite( "SobelYDirection.jpg", ySobel );
    imwrite( "SobelMag.jpg", magSobel );
    imwrite( "SobelDir.jpg", dirSobel );

    Mat resultX(image.rows,image.cols, CV_8UC1);
    Mat resultY(image.rows,image.cols, CV_8UC1);
    Mat resultMag(image.rows,image.cols, CV_8UC1);
    Mat resultDir(image.rows,image.cols, CV_8UC1);

    normalize(xSobel,resultX,0,255,NORM_MINMAX, CV_8UC1);
    normalize(ySobel,resultY,0,255,NORM_MINMAX, CV_8UC1);
    normalize(magSobel,resultMag,0,255,NORM_MINMAX);
    normalize(dirSobel,resultDir,0,255,NORM_MINMAX);
    imwrite("grad_x.jpg",resultX);
    imwrite("grad_y.jpg",resultY);
    imwrite("mag.jpg",resultMag);
    imwrite("dir.jpg", resultDir);

    int ***houghSpace = malloc3dArray(image.rows, image.cols, MAXRADIUS);

    Mat testing = imread("mag.jpg", 1);
    Mat gray_test;
    cvtColor( testing, gray_test, CV_BGR2GRAY );

    Mat thresh = threshold(gray_test);

    hough(image, thresh, dirSobel, houghSpace, MAXRADIUS-MINRADIUS);

    return 0;
}

void Sobel(cv::Mat &input, int size, cv::Mat &magOutput, cv::Mat &xOutput, cv::Mat &yOutput, cv::Mat &dirOutput) {

    int kernelX[3][3] = {{-1, 0, 1},
                            {-2, 0, 2},
                            {-1, 0, 1}};

    int kernelY[3][3] = {{-1, -2, -1},
                            {0, 0, 0},
                            {1, 2, 1}};

    cv::Mat kernelXMat = cv::Mat(size, size, cv::DataType<int>::type, kernelX);
    cv::Mat kernelYMat = cv::Mat(size, size, cv::DataType<int>::type, kernelY);

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
			float sumX = 0.0;
            float sumY = 0.0;

			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					int kernalvalX = kernelXMat.at<int>(kernelx, kernely);
                    int kernalvalY = kernelYMat.at<int>(kernelx, kernely);

					// do the multiplication
					sumX += imageval * kernalvalX;		
                    sumY += imageval * kernalvalY;					
				}
			}
			// set the output value as the sum of the convolution
			xOutput.at<float>(i, j) = (float) sumX;
            yOutput.at<float>(i, j) = (float) sumY;

            magOutput.at<float>(i, j) = (float) sqrt((sumX * sumX) + (sumY * sumY));

            dirOutput.at<float>(i, j) = (float) atan2(sumY, sumX);
		}
	}
}

Mat threshold(cv::Mat &input) {
    Mat thresh(input.rows, input.cols, CV_8UC1);

    for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
            int imageval = ( int ) input.at<uchar>( i, j );
            
            if (imageval > 60) {
                thresh.at<uchar>( i, j ) = 255;
            } else {
                thresh.at<uchar>( i, j ) = 0;
            }
        }
    }
    imwrite("thresh.jpg", thresh);

    return thresh;
}

void hough(Mat &input, Mat &gradient, Mat &direction, int ***houghSpace, int radius) {
    for (int i = 0; i < gradient.rows; i++) {
        for (int j = 0; j < gradient.cols; j++) {
            for (int k = 0; k < MAXRADIUS; k++) {
                houghSpace[i][j][k] = 0;
            }
        }
    }


    for (int x = 0; x < gradient.rows; x++) {
        for (int y = 0; y < gradient.cols; y++) {
            if (gradient.at<uchar>(x, y) == 255) {
                for (int r = MINRADIUS; r < MAXRADIUS; r++) {
                    int b =  y - (r * cos(direction.at<float>(x, y)));
                    int a =  x - (r * sin(direction.at<float>(x, y)));

                    int d =  y + (r * cos(direction.at<float>(x, y)));
                    int c =  x + (r * sin(direction.at<float>(x, y)));
                    
                    if (b >= 0 && b < gradient.cols && a >= 0 && a < gradient.rows) {
                        houghSpace[a][b][r] += 1;
                    }
                    if (d >= 0 && d < gradient.cols && c >= 0 && c < gradient.rows) {
                        houghSpace[c][d][r] += 1;
                    }
                }
            }
        }
    }

    Mat houghSpaceOutput(gradient.rows, gradient.cols, CV_32FC1);

    for (int x = 0; x < gradient.rows; x++) {
        for (int y = 0; y < gradient.cols; y++) {
            for (int r = MINRADIUS; r < MAXRADIUS; r++) {
                houghSpaceOutput.at<float>(x,y) += houghSpace[x][y][r];
                if (houghSpace[x][y][r] > 20) {
                    std::cout << "circle" << std::endl;
                    circle(input, Point(y, x), r, (0, 255, 255), 2);
                }
                // if(!(pow((x-xc),2) + pow((y-yc),2) > pow(rc,2))) {
                //     test_pass = false;
                // }
            }
            
        }
    }

    Mat houghSpaceConvert(gradient.rows, gradient.cols, CV_8UC1);
    normalize(houghSpaceOutput, houghSpaceConvert, 0, 255, NORM_MINMAX);

    imwrite( "houghOuput.jpg", houghSpaceConvert );
    imwrite("output3.jpg", input);
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D 
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);
	
	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{	
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;							
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}
