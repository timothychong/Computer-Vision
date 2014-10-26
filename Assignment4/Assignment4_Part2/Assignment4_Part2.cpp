/**
	CS585_Assignment4.cpp
	@author: 
	@version: 

	CS585 Image and Video Computing Fall 2014
	Assignment 4: AQUARIUM
*/

#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>
#include "filterBundle.h"
#include <limits>

using namespace cv;
using namespace std;

/**
TODO: Read the segmented images of the aquarium dataset and create the following:
	a) A binary image where background pixels are labelled 0, and object pixels are
		labelled 255
	b) A labelled image where background pixels are labelled 0, and object pixels are
		labelled numerically 
	c) A 3-channel image where the object centroids are drawn as red dots on the binary image
*/

void readImages(String str, String extend, vector<String> &fileVector, int start_frame, int number_of_frames, int zeros);
/*
This function determines the name of the images to be read
*/

void semgent2binary(vector<Mat> & images, vector<Mat> & segmented, vector<vector<Point>> & centroids);
/*
This function does the segmentation of the original image input into a set of binary objects, where
each object has its own number (non-zero). Only run for the fish data.
*/



int main()
{
	// what sequence are we loading?
	String select = "fish";

	// first, determine number of frames to read in based on sequence we are talking about
	int start_frame,number_of_frames, zeros;
	String baseName, extend;
	if (select=="fish"){
		start_frame = 1;
		number_of_frames = 53-1;
		zeros = 2; // it ranges from frame 01 to 53, 2 digits
		baseName = "Segmentation_Aqua/2014_aq_segmented ";
		extend = ".jpg";
	}
	else if (select=="bats"){
		start_frame = 750;
		number_of_frames = 10; // it's more than this.. change eventually
		zeros = 3; // it ranges from 750 to 900, 3 digits
		baseName = "Segmentation_Bats/CS585Bats-Segmentation_frame000000";
		extend = ".txt";
	}

	// read in all the necessary files, add them to an array:
	vector<String> imagenames;
	vector<Mat> images;
	readImages(baseName, extend, imagenames, start_frame, number_of_frames, zeros);
	for (int i=0; i < imagenames.size(); i++){
		images.push_back(imread(imagenames[i]));
	}

	// now run segmentation on all the frames, return multi-obj binary images and centroids
	vector<Mat> segmented;
	vector<vector<Point>> centroids;
	semgent2binary(images, segmented, centroids);
	
	//cout << "size segmented: " << segmented.size() << endl;
	//cout << "size centroids: " << centroids.size() << endl;
	// save the binary images, for reference
	
	cout << "Writing binary images (segmented)" << endl;
	for (int i=1; i< segmented.size()+1; i++){
		
		// write out the image file
		String fileo = "binary_Aqua/binaryAqua_";
		int digit = (i == 0)? 2 : zeros - 1 - (int)log10(i);
		
		ostringstream vv;
		vv << i;
		fileo.append(string(digit, '0'));
		fileo.append(string(vv.str()));
		fileo.append(string(".jpg"));

		
		// draw on the centroid for saving purposes
		Mat output;// = Mat::zeros(segmented[i].size(), CV_8UC3);
		cvtColor(segmented[i-1],output,CV_GRAY2BGR);
		
		// draw the centroids
		for (int j=0; j<centroids[i-1].size(); j++){
			circle(output, centroids[i-1][j], 2, Scalar(0,0,255), -1, 8);
		}
        imwrite(fileo, output);
	}

	
	// now run the tracking process


	
	char key = waitKey(0);

	return 0;
}


// function for reading frames into a vector
void readImages(String str, String extend, vector<String> &fileVector, int start_frame, int number_of_frames, int zeros){
	// assumes 3 digit numbers only, larger will break it.
	int number_of_digits = zeros;
	String cpy = str;
	for ( int i= start_frame; i <  start_frame + number_of_frames; i ++ ) {
        
        int digit = (i == 0)? 2 : number_of_digits - 1 - (int)log10(i);
		str = cpy; //reset to original string name
        ostringstream ss;
        ss << i;

        str.append(string(digit, '0'));
        str.append(string(ss.str()));
        str.append(extend);

        fileVector.push_back(str);
		cout << "image file: " << str << endl;
    }

}

void semgent2binary(vector<Mat> & images, vector<Mat> & segmented, vector<vector<Point>> & centroids){
	//namedWindow("window");

	for (int im=0; im<images.size(); im++){
		//namedWindow("window");
		imshow("original",images[im]);
		Mat binary = Mat::zeros(images[im].size(),CV_8UC1);

		
		// first color red to white, and everything else stays black
		for (int x=0; x< images[im].rows; x++){
			for (int y=0; y< images[im].cols; y++){
				Vec3b color = images[im].at<Vec3b>(x,y);
				// thresholding to get only red of fish, and not any of the green
				if (images[im].at<Vec3b>(x,y)[2] > 70 && images[im].at<Vec3b>(x,y)[1] < 120){
					binary.at<uchar>(x,y) = 255;
					//cout << "here!!" << endl;
				}
			}
		}

		// dilate and erode it slightly for better fills
		int erosion_size = 1;
		int dilation_size = 1;
		Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
		
		dilate(binary, binary, element);
		erode(binary, binary, element);
		
		imshow("segmented",binary);

		// now run contours (fill with according color)
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		Mat contour_output = Mat::zeros(binary.size(), CV_8UC1);
		
		//decent, but perhaps use canny edges here
		//int thresh = 100;
		//Canny( binary, binary, thresh, thresh*2, 3);
		
		// find and fill each contour with an increasing number
		findContours(binary,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
		for (int i=0; i< contours.size(); i++){
			drawContours(contour_output, contours, i, 255, CV_FILLED, 8, hierarchy);
		}
		cout << "no. first contours: " << contours.size() << endl;

		// run contours again to clean up geometry and other conflicts (e.g. before had 79 contours, after 38)
		Mat finalSegmented = Mat::zeros(binary.size(),CV_8UC1);
		// make new vector for contours and hierarchy..?
		findContours(contour_output,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
		cout << "no. second contours: " << contours.size() << endl;
		vector<Point> centroidVec;
		for (int i=0; i< contours.size(); i++){

			// fill the object with the according color
			//drawContours(contour_output, contours, i, i*2+50, CV_FILLED, 8, hierarchy);
			drawContours(finalSegmented, contours, i, i*4+50, CV_FILLED, 8, hierarchy);

			//push on the centroid
			Moments m = moments(contours[i], true);
			Point centroid(m.m10/m.m00,m.m01/m.m00);
			centroidVec.push_back(centroid);
			// does not check for case of "bad centroid", e.g. negative or infs (would happen with first contour pass)
			//circle(finalSegmented, centroid, 2, 255, -1, 8);

			
		}

		centroids.push_back(centroidVec);
		segmented.push_back(finalSegmented.clone());
		//imshow("contour output", finalSegmented);
		
	}
	
	// all done!
	cout << "segmented image" << endl;

}