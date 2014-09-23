/**
	CS585_Assignment2_part1.cpp
	@author: Timothy Chong & Patrick W. Crawford

	CS585 Image and Video Computing Fall 2014
	Assignment 2
	--------------
	This program:
		a) (Part 1) Tracks an object by template matching
		b) (Part 2) Recognizes hand shapes or gestures and creates a graphical display
	--------------
	PART A
*/

#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

void addBoundingBox(Mat &src, Point p, int width, int height);
void correlation(Mat & src, Mat & templ, Mat & dest, Point & max, float & maxIntensity);

int main()
{
	VideoCapture cap(0);

	// if not successful, exit program
    if (!cap.isOpened())
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }

    Mat corr_mat;
    Mat templ;
    vector<Mat> resized_frame;
	
	// add in empty mats to place in the results of pyramid template matching
    for (int i = 0; i < 5; i ++ ) {
        resized_frame.push_back(Mat());
    }
	
	// load in the template for matching
    templ = imread("template.PNG", 3);

	namedWindow("MyVideo",WINDOW_AUTOSIZE);
    imshow("Correlation", templ);
    moveWindow("MyVideo", 0,0);

	// clocks for timing purposes
	 clock_t t1,t2;

	while (1)
    {
		
		//timing start
		t1=clock();
		
		// read a new frame from video
        bool bSuccess = cap.read(resized_frame[0]);
		
		// crate multiple scaled down versions of the source image for matching purposes
		// our pyramid  uses 5 levels, so there are 4 scaled down images + the original
        Size s = resized_frame[0].size();
        for (int i = 1; i < 5; i++) {
            s.height *= 0.75;
            s.width *= 0.75;
            resize(resized_frame[0], resized_frame[i], Size(s.width, s.height), INTER_CUBIC);
        }

		//if not successful, break loop
        if (!bSuccess)
        {
             cout << "Cannot read a frame from video stream" << endl;
             break;
        }
		
		// pre-allocate some essential variables for best matching logic
        Point p = Point(-1, -1);
        Point finalP;
        float max_intensity = 0;
        float max_index = -1;
        float temp;
        
        // run the correlation on each frame
        for (int i = 0; i < 5; i++) {
            p = Point(-1, -1);
            correlation(resized_frame[i], templ, corr_mat, p, temp);
            if (temp > max_intensity) {
                max_intensity = temp;
                max_index = i;
                finalP = p;
            }
        }
        
        // if the object has been found, draw a box
        if (max_index != -1){
        	// calculate the size of the box from the index best matched
            double factor = pow(0.75, max_index);
            // add the box
            addBoundingBox(resized_frame[0], Point(finalP.x / factor, finalP.y / factor) , templ.cols / factor, templ.rows / factor);
        }
		imshow("MyVideo", resized_frame[0]);
        imshow("Correlation", corr_mat);

		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}

		// stuff for clock timings
		t2=clock();
		float diff ((float)t2-(float)t1);
		cout<< "Time difference: " << diff<<endl;


	}

	cap.release();
	return 0;
}


// This function gets the brightest, best correlating position after template matching
void correlation(Mat & src, Mat & templ, Mat & dest, Point & max, float & maxIntensity) {
    // use the built in template matching function
    matchTemplate( src, templ, dest, CV_TM_CCOEFF_NORMED);
	
	// get the brightest pixel, representing the best matched location
    maxIntensity = 0;
    float temp = 0;
    for(int x = 0; x < dest.cols; x ++ ) {
        for(int y = 0; y < dest.rows; y ++ ) {
            temp = dest.at<float>(y, x);
            if (temp > maxIntensity && temp > 0.6) {
                maxIntensity = temp;
                max = Point(x, y);
            }
        }
    }
}

// this function adds the bounding box around the object, given the center
// width and height. It modifies the original image.
void addBoundingBox(Mat &src, Point p, int width, int height) {

    if (p.x == -1) return;
	// set color of bounding box, in this case red out of BGR
    Vec3b color = Vec3b(0,0,255);
	
	// logic of the indices for where the bounds of the lines are
    int leftBound = p.x;
    leftBound = (leftBound < 0)? 0 : leftBound;
    int rightBound = p.x + width;
    rightBound = (rightBound >= src.cols)? src.cols - 1 : rightBound;
    int lowerBound = p.y;
    lowerBound = (lowerBound < 0)? 0 : lowerBound;
    int upperBound = p.y + height;
    upperBound = (upperBound >= src.rows)? src.rows - 1 : upperBound;
	
	// add the horizontal bars
    for(int i = leftBound; i < rightBound; i++) {
        src.at<Vec3b>(upperBound, i) = color;
        src.at<Vec3b>(lowerBound, i) = color;
    }
	
	// add the vertical bars
    for(int i = lowerBound; i < upperBound; i++) {
        src.at<Vec3b>(i, leftBound) = color;
        src.at<Vec3b>(i, rightBound) = color;
    }
}

