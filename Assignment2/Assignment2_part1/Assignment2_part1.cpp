/**
	CS585_Assignment2.cpp
	@author:
	@version:

	CS585 Image and Video Computing Fall 2014
	Assignment 2
	--------------
	This program:
		a) Tracks an object by template matching
		b) Recognizes hand shapes or gestures and creates a graphical display
	--------------
	PART A
*/

#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

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

	Mat frame;
    Mat corr_mat;
    Mat templ;

    templ = imread("template.PNG", 3);

	namedWindow("MyVideo",WINDOW_AUTOSIZE);
	namedWindow("Correlation",WINDOW_AUTOSIZE);
    imshow("Correlation", templ);

	while (1)
    {
		// read a new frame from video
        bool bSuccess = cap.read(frame);

		//if not successful, break loop
        if (!bSuccess)
        {
             cout << "Cannot read a frame from video stream" << endl;
             break;
        }

        Point p = Point(-1, -1);
        float intensity;
        correlation(frame, templ, corr_mat, p, intensity);
        addBoundingBox(frame, p, templ.cols, templ.rows);
		imshow("MyVideo", frame);
        imshow("Correlation", corr_mat);

		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}

	}

	cap.release();
	return 0;
}


void correlation(Mat & src, Mat & templ, Mat & dest, Point & max, float & maxIntensity) {
    matchTemplate( src, templ, dest, CV_TM_CCOEFF_NORMED);

    maxIntensity = 0;
    float temp = 0;
    for(int x = 0; x < dest.cols; x ++ ) {
        for(int y = 0; y < dest.rows; y ++ ) {
            temp = dest.at<float>(y, x);
            if (temp > maxIntensity && temp > 0.5) {
                maxIntensity = temp;
                max = Point(x, y);
            }
        }
    }
   if (maxIntensity > 0.5) {
        cout << max << endl;
   }
}

void addBoundingBox(Mat &src, Point p, int width, int height) {

    if (p.x == -1) return;

    Vec3b color = Vec3b(0,0,255);

    int leftBound = p.x;
    leftBound = (leftBound < 0)? 0 : leftBound;
    int rightBound = p.x + width;
    rightBound = (rightBound >= src.cols)? src.cols - 1 : rightBound;
    int lowerBound = p.y;
    lowerBound = (lowerBound < 0)? 0 : lowerBound;
    int upperBound = p.y + height;
    upperBound = (upperBound >= src.rows)? src.rows - 1 : upperBound;

    for(int i = leftBound; i < rightBound; i++) {
        src.at<Vec3b>(upperBound, i) = color;
        src.at<Vec3b>(lowerBound, i) = color;
    }

    for(int i = lowerBound; i < upperBound; i++) {
        src.at<Vec3b>(i, leftBound) = color;
        src.at<Vec3b>(i, rightBound) = color;
    }
}

