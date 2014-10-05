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

int main(int argc, char **argv)
{
	// read image as grayscale
    Mat img = imread("circle.jpg"); if(!img.data) {
        cout << "File not found" << std::endl;
        return -1;
    }

    Mat im_gray;
    Mat binary;

    cvtColor(img.clone(),im_gray,CV_RGB2GRAY);

	// invert the image
    bitwise_not ( im_gray, im_gray );

    convertToBinary(im_gray.clone(), binary);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    int maxsize = 0;
    int maxind = 0;

    findBiggestContour (binary, contours, hierarchy, maxsize, maxind);
    Mat contour_output = Mat::zeros( binary.size(), CV_8UC3 );;

	//// draw filled in largest object in black
    drawContours(contour_output, contours, maxind, 	Scalar::all(255), CV_FILLED, 8, hierarchy);

    // get information about the object:
    int area = maxsize;
    double orientation = getOrientation(contours[maxind]); // angle that the axis of least inertia makes with x-axis
    double circulatiry = calculateCircularity (contour_output, contours[maxind]) ; // Emin/Emax
    double perimeter = arcLength(contours[maxind],1); // perimeter of largest contour
    double euler = calculateEulerNumber(binary);	// euler number, which in the way we are doing it now should always be 1 (one blob, no holes)

    cout << endl;
    cout << "Area: " << area << " units squared" << endl;
    cout << "Orientation: " << orientation << " degrees" << endl;
    cout << "Circularity: " << circulatiry << " units" << endl;
    cout << "Length of Perimeter: " << perimeter << " units" << endl;
    cout << "Compactness: " << perimeter*perimeter/area << " units" << endl;
    cout << "Euler number: " << euler << " units" << endl;

	//// create windows
    namedWindow("Thresh");
    namedWindow("Contour");
    moveWindow("Thresh", 30,190);
    moveWindow("Contour", 760,190);
    imshow("Original", img);	// note img_bw is inverted here, as original processing made the forground white
    imshow("Thresh",binary);	// note img_bw is inverted here, as original processing made the forground white
    imshow("Contour",contour_output);

	//show the binary image, as well as the labelled image
    waitKey();
    return 0;
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
     for(vector<Vec4i>::size_type idx=0; idx<hierarchy.size(); ++idx)
    {
        if(hierarchy[idx][3] != -1){
            holes ++;
        }
    }
    imshow("Original",src);

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
