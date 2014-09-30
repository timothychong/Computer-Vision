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
    Mat img = imread("blob.png", 0); 
    if(!img.data) {
        cout << "File not found" << std::endl;
        return -1;
    }

	// create windows
	namedWindow("Original");
    namedWindow("binary");
    namedWindow("labelled");
	imshow("Original",img);
	// perform morphological operations to remove small objects (noise) and fill black spots inside objects
	// %TODO
	// Documentation on erode: http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=erode#erode
	// Documentation on dilate: http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=erode#dilate
	
	// initialize structuring element for morphological operations
	int erosion_size = 3;
	int dilation_size = 3;
	Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
	//perform erosions and dilations
	erode(img, img, element);
	dilate( img, img, element );
	dilate( img, img, element );

	//convert thresholded image to binary image
    Mat binary;
    threshold(img, binary, 0.0, 1.0, THRESH_BINARY);

	//initialize vector to store all blobs
	vector <vector<Point2i>> blobs;

	//call function that detects blobs and modifies the input binary image to store blob info
    FindBinaryLargeObjects(binary, blobs);

	//display the output
	Mat output = Mat::zeros(img.size(), CV_8UC3);
    // Randomly color the blobs
    for(size_t i=0; i < blobs.size(); i++) {
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));

        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;

            output.at<Vec3b>(y,x)[0] = b;
            output.at<Vec3b>(y,x)[1] = g;
            output.at<Vec3b>(y,x)[2] = r;
        }
    }

	//show the binary image, as well as the labelled image
    imshow("binary", img);
    imshow("labelled", output);
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

