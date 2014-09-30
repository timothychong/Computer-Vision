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
#include <math.h>

using namespace cv;
using namespace std;

/**
	Function that detects blobs from a binary image
	@param binaryImg The source binary image (binary image contains pixels labeled 0 or 1 (not 255))
	@param blobs Vector to store all blobs, each of which is stored as a vector of 2D x-y coordinates
*/
//void FindBinaryLargeObjects(const Mat &binaryImg, vector <vector<Point2i>> &blobs);
void FindBinaryLargeObjects(const Mat &binary, vector <vector<Point2i>> &blobs);

int main(int argc, char **argv)
{
	// read image as grayscale
    Mat img = imread("bcancer2.png");
    if(!img.data) {
        cout << "File not found" << std::endl;
        return -1;
    }

	Mat img_cp = img.clone();
	Mat im_gray;
	cvtColor(img_cp,im_gray,CV_RGB2GRAY);

	// invert the image
	im_gray = 255-im_gray;
	

	//blur( img, img, Size(5,5) );
    Mat img_bw;
	int erosion_size = 1;
	int dilation_size = 1;
	Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

	//perform erosions and dilations and  blur
	blur( im_gray, im_gray, Size(5,5) );
    erode(im_gray, im_gray, element);
    dilate( im_gray, im_gray, element );
	

	// prior to invert
	//threshold(im_gray, img_bw, 220, 255, THRESH_BINARY);
	threshold(im_gray, img_bw, 40, 255, THRESH_BINARY);
	
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
	// turn the image white
	contour_output.setTo(Scalar(255,255,255));


	// draw filled in largest object in black
    drawContours(contour_output, contours, maxind, 	Scalar(0, 0, 0), CV_FILLED, 8, hierarchy, 0);
	//drawContours(contour_output, contours, maxind, Scalar(0,0,255), 2, 8, hierarchy);
	
	
	// get information about the object:
	int area = maxsize;
	double orientation = -1; // angle that the axis of least inertia makes with x-axis
	int circulatiry = -1; // Emin/Emax
	double perimeter = arcLength(contours[maxind],1); // perimeter of largest contour
	int euler = -99999;	// euler number, which in the way we are doing it now should always be 1 (one blob, no holes)

	// find the second moment to get orientation and circulatiry
	
	// calculations based on the paper from class, Computer Vision for Interactive Computer Graphics, page 43
	Moments mo = moments(contours[maxind],false);
	double xc = mo.m10/mo.m00;
	double yc = mo.m01/mo.m00;
	double a = mo.m20/mo.m00 - xc*xc;
	double b = 2*(mo.m11/mo.m00 - xc*yc);
	double c = mo.m02/mo.m00 - yc*yc;
	orientation = atan2(b,(a-c))/2 * (180 / 3.1415926); // output in degrees


	cout << endl;
	cout << "Area: " << area << " units squared" << endl;
	cout << "Orientation: " << orientation << " degrees" << endl;
	cout << "Circularity: " << circulatiry << " units" << endl;
	cout << "Length of Perimeter: " << perimeter << " units" << endl;
	cout << "Compactness: " << perimeter*perimeter/area << " units" << endl;
	cout << "Euler number: " << euler << " units // IMPLEMENT!" << endl;

	// create windows
	namedWindow("Thresh");
	namedWindow("Contour");
	moveWindow("Thresh", 30,190);
	moveWindow("Contour", 760,190);
    imshow("Thresh",255-img_bw);	// note img_bw is inverted here, as original processing made the forground white
    imshow("Contour",contour_output);

	//show the binary image, as well as the labelled image
    waitKey(0);
    return 0;
}


void FindBinaryLargeObjects(const Mat &binary, vector <vector<Point2i>> &blobs)
{
	//clear blobs vector
    blobs.clear();

	//labelled image of type CV_32SC1
	Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

	//label count starts at 2
    int label_count = 2; 

	//iterate over all pixels until a pixel with a 1-value is encountered
    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
			
            if(row[x] != 1) {
                continue;
            }

			//floodFill the connected component with the label count
			//floodFill documentation: http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
            Rect rect;
            floodFill(label_image, Point(x,y), label_count, &rect, 0, 0, 4);

			//store all 2D co-ordinates in a vector of 2d points called blob
            vector <Point2i> blob;
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }
                    blob.push_back(Point2i(j,i));
                }
            }
			//store the blob in the vector of blobs
            blobs.push_back(blob);

			//increment counter
            label_count++;
        }
    }
	//cout << "The number of blobs in the image is: " << label_count;
	//Code derived from: http://nghiaho.com/
}
