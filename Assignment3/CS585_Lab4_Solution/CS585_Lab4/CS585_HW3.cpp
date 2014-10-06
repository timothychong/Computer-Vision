/**
	CS585_Lab4.cpp
	@author: Ajjen Joshi
	@version: 1.0 9/25/2014

	CS585 Image and Video Computing Fall 2014
	Lab 4
	--------------
	This program introduces the following concepts:
		a) Understanding and applying basic morphological operations on images
		b) Finding and labeling blobs in a binary image
	--------------
*/


#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

/**
	Function that detects blobs from a binary image
	@param binaryImg The source binary image (binary image contains pixels labeled 0 or 1 (not 255))
	@param blobs Vector to store all blobs, each of which is stored as a vector of 2D x-y coordinates
*/
void convertToBinary (Mat &src, Mat &dst);
void findBiggestContour (Mat &, vector<vector<Point>> &, vector<Vec4i> &, int & , int & );
int calculateEulerNumber(Mat &);
double getOrientation (vector<Point>);
double calculateCircularity (Mat & src, vector<Point> & contour);
void part3(Mat & src);
void thinningIteration(Mat& im, int iter);
void thinning(Mat& im);
double* part1(Mat & img, bool);
void part4(Mat & img);

int main(int argc, char **argv)
{

	// read image as grayscale
    Mat img = imread("bcancer3.0.png"); if(!img.data) {
        cout << "File not found" << std::endl;
        return -1;
    }


    part1(img,  true);
    //part3(img);
    //part4(img);
    namedWindow("Original");
    imshow("Original", img);	// note img_bw is inverted here, as original processing made the forground white

    waitKey();
    return 0;
}


void part4(Mat & img) {

    vector<Mat> images;
    images.push_back(imread("bcancer1.png"));
    images.push_back(imread("bcancer2.png"));
    images.push_back(imread("bcancer3.png"));
    images.push_back(imread("bcancer4.png"));

    vector<double *> results;
    for (int i = 0; i < images.size(); i++ ) {
        results.push_back(part1(images[i], false));
    }
    //for (int i = 0; i < images.size(); i++ ) {
        //for(int a = 0; a < 2; a++) {
            //cout << images[i][a] << " " ;
        //}
        //cout << endl;
    //}

}


double * part1(Mat & img, bool showWindow = true){
    Mat im_gray;
    Mat binary;

    cvtColor(img.clone(),im_gray,CV_RGB2GRAY);

	//// invert the image
    bitwise_not ( im_gray, im_gray );
    convertToBinary(im_gray.clone(), binary);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    int maxsize = 0;
    int maxind = 0;

    findBiggestContour (binary, contours, hierarchy, maxsize, maxind);
    Mat contour_output = Mat::zeros( binary.size(), CV_8UC3 );

	////// draw filled in largest object in black
    drawContours(contour_output, contours, maxind, 	Scalar::all(255), CV_FILLED, 8, hierarchy);

    //// get information about the object:
    int area = maxsize;
    double orientation = getOrientation(contours[maxind]); // angle that the axis of least inertia makes with x-axis
    double circularity = calculateCircularity (contour_output, contours[maxind]) ; // Emin/Emax
    double perimeter = arcLength(contours[maxind],1); // perimeter of largest contour
    double euler = calculateEulerNumber(binary);// euler number, which in the way we are doing it now should always be 1 (one blob, no holes)

    cout << endl;
    cout << "Area: " << area << " units squared" << endl;
    cout << "Orientation: " << orientation << " degrees" << endl;
    cout << "Circularity: " << circularity << " units" << endl;
    cout << "Length of Perimeter: " << perimeter << " units" << endl;
    cout << "Compactness: " << perimeter*perimeter/area << " units" << endl;
    cout << "Euler number: " << euler << " units" << endl;

    if (showWindow){
        //// create windows
        namedWindow("Thresh");
        namedWindow("Contour");

        moveWindow("Thresh", 30,190);
        moveWindow("Contour", 760,190);
        imshow("Thresh",binary);	// note img_bw is inverted here, as original processing made the forground white
        imshow("Contour",contour_output);
    }

    double * result = new double[5];
    result[0] = area;
    result[1] = orientation;
    result[2] = circularity;
    result[3] = perimeter;
    result[4] = euler;
    return result;
}


int calculateEulerNumber(Mat &src)
{
    int holes = 0;
    int bodies = 0;

     vector<vector<Point>> contours;
     vector<Vec4i> hierarchy;

    findContours( src.clone(), contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    bodies = contours.size();

    findContours( src.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );

    holes = 0;
     for(int idx=0; idx<hierarchy.size(); ++idx)
    {
        if(hierarchy[idx][3] != -1){
            holes ++;
        }
    }
    imshow("Original", src);

    return bodies - holes;
}

void convertToBinary (Mat &src, Mat &dst) {

    blur( src, src, Size(5,5) );
    int erosion_size = 1;
    int dilation_size = 1;
    Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

	////perform erosions and dilations and  blur
    blur( src, src, Size(5,5) );
    erode(src, src, element);
    dilate( src, src, element );

	//// prior to invert
    threshold(src, dst, 40, 255, THRESH_BINARY);

}

void findBiggestContour (Mat & src, vector<vector<Point>> & contours, vector<Vec4i> & hierarchy, int & maxsize, int & maxind ){

    findContours(src.clone(), contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    for(unsigned int i = 0; i < contours.size(); i++ )
    {
        // Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
        double area = contourArea(contours[i]);
        if (area > maxsize) {
            maxsize = (int) area;
            maxind = i;
        }
    }
}


double getOrientation (vector<Point> contour) {
    Moments mo = moments(contour,false);
    double xc = mo.m10/mo.m00;
    double yc = mo.m01/mo.m00;
    double a = mo.m20/mo.m00 - xc*xc;
    double b = 2*(mo.m11/mo.m00 - xc*yc);
    double c = mo.m02/mo.m00 - yc*yc;
    return atan2(b,(a-c))/2 * (180 / 3.1415926); // output in degrees
}

double calculateCircularity (Mat & src, vector<Point> & contour) {

    double a = 0, b = 0, c = 0;

    Moments m = moments(contour, true);
    Point centroid(m.m10/m.m00, m.m01/m.m00);

    for(int x = 0; x < src.rows; x ++ ) {
        for(int y = 0; y < src.cols; y ++ ) {
            if (src.at<Vec3b>(x,y)[0] != 0){
                a += (x - centroid.y) * (x - centroid.y);
                b += (x - centroid.y) * (y - centroid.x);
                c += (y - centroid.x) * (y - centroid.x);
            }
        }
    }
    b *= 2;
    double first_term = ((a - c)/2) * ((a - c) / sqrt((a - c) * (a - c) + b * b));
    double double_term = (b/2) * (b / sqrt((a - c) * (a - c) + b * b));

    double emin = (a + c) /2 - first_term - double_term;
    double emax = (a + c) /2 + first_term + double_term;

    return emin / emax;

}

void part3(Mat & src){

    Mat dst;
    Mat channel[3];
    split(src, channel);
    blur( channel[0], dst, Size(4,4) );

    double average = 0;
    for(int x = 0; x < src.rows; x ++ ) {
        for(int y = 0; y < src.cols; y ++ ) {
            average += src.at<Vec3b>(x,y)[0];
        }
    }

    average /= (src.rows * src.cols);

    threshold(dst, dst, average,  0, THRESH_TOZERO_INV);


    int erosion_size = 4;
    int dilation_size = 4;


    Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

    //perform erosions and dilations
    erode(dst, dst, element, Point(-1, -1), 2);
    dilate(dst, dst, element);

    vector<vector<Point>> contours;
     vector<Vec4i> hierarchy;

    findContours( dst.clone(), contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

    dst = Mat::zeros( src.size(), CV_8UC1 );

    for(int i = 0; i < contours.size(); i++) {
        RotatedRect rect = minAreaRect( Mat(contours[i] ));
        Size size = rect.size;

        if ((contourArea(contours[i]) > 10000) && (size.height > (3 * size.width) || size.width > (3 * size.height)))  {
            drawContours(dst, contours, i, 	Scalar::all(255), CV_FILLED, 8, hierarchy);
        }
    }


    Mat img = dst.clone();

    thinning(img);

    namedWindow("Hi");
    imshow("Hi", img);
    imwrite("abc.png", img);	// note img_bw is inverted here, as original processing made the forground white

}

//http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/
void thinningIteration(Mat& im, int iter)
{
    Mat marker = Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinning(Mat& im)
{
    im /= 255;

    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (countNonZero(diff) > 0);

    im *= 255;
}




