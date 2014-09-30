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
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/**
	Function that detects blobs from a binary image
	@param binaryImg The source binary image (binary image contains pixels labeled 0 or 1 (not 255))
	@param blobs Vector to store all blobs, each of which is stored as a vector of 2D x-y coordinates
*/
void FindBinaryLargeObjects(const Mat &binaryImg, vector <vector<Point2i>> &blobs);

int main(int argc, char **argv)
{
	// read image as grayscale
    Mat img = imread("bcancer1.png");
    if(!img.data) {
        cout << "File not found" << std::endl;
        return -1;
    }
    //Converting to grayscale
    Mat im_gray;
    cvtColor(img, im_gray,CV_RGB2GRAY);

	blur( img, img, Size(5,5) );
    Mat img_bw;
    threshold(im_gray, img_bw, 220.0, 255.0, THRESH_BINARY);


	int erosion_size = 2;
	int dilation_size = 2;
	Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
	//perform erosions and dilations
    erode(img_bw, img_bw, element);
    dilate( img_bw, img_bw, element );

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
    Mat temp = img_bw.clone();
    findContours(temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


	cout << "The number of contours detected is: " <<  contours.size() << endl;

	int maxsize = 0;
	int maxind = 0;
	Rect boundrec;
	for(int i = 0; i < contours.size(); i++ )
	{
		// Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
		double area = contourArea(contours[i]);
		if (area > maxsize) {
			maxsize = area;
			maxind = i;
			boundrec = boundingRect(contours[i]);
		}
	}

	Mat contour_output = Mat::zeros( img.size(), CV_8UC3 );
    drawContours(contour_output, contours, maxind, 	Scalar(255, 0, 0), CV_FILLED, 8, hierarchy, 0);
	//drawContours(contour_output, contours, maxind, Scalar(0,0,255), 2, 8, hierarchy);
    //rectangle(contour_output, boundrec, Scalar(0,255,0),1, 8,0);

	// create windows
	namedWindow("Original");
	namedWindow("Contour");
    imshow("Original",img_bw);
    imshow("Contour",contour_output);

	//show the binary image, as well as the labelled image
    waitKey(0);
    return 0;
}
