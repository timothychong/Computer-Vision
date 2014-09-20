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

void correlation(Mat & src, Mat & templ, Mat & dest);
void addBoundingBox(Mat &src, Point p, int width, int height);

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

        Point p = Point(10,10);
        correlation(frame, templ, corr_mat);
        addBoundingBox(frame, p, 50, 100);
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


Point correlation(Mat & src, Mat & templ, Mat & dest) {
    matchTemplate( src, templ, dest, CV_TM_CCOEFF_NORMED);

}

void addBoundingBox(Mat &src, Point p, int width, int height) {

    Vec3b color = Vec3b(0,0,255);

    int leftBound = (p.x - width/2);
    leftBound = (leftBound < 0)? 0 : leftBound;
    int rightBound = (p.x + width/2);
    rightBound = (rightBound >= src.cols)? src.cols - 1 : rightBound;
    int lowerBound = p.y - height/2;
    lowerBound = (lowerBound < 0)? 0 : lowerBound;
    int upperBound = p.y + height/2;
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

