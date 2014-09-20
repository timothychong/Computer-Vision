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
#include <math.h>
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

    Mat corr_mat;
    Mat templ;
    vector<Mat> resized_frame;

    for (int i = 0; i < 5; i ++ ) {
        resized_frame.push_back(Mat());
    }

    templ = imread("template.PNG", 3);

	namedWindow("MyVideo",WINDOW_AUTOSIZE);
    //namedWindow("MyVideo1",WINDOW_AUTOSIZE);
    //namedWindow("MyVideo2",WINDOW_AUTOSIZE);
    //namedWindow("MyVideo3",WINDOW_AUTOSIZE);
    //namedWindow("MyVideo4",WINDOW_AUTOSIZE);
    //namedWindow("Correlation",WINDOW_AUTOSIZE);
	//namedWindow("Red Video",WINDOW_AUTOSIZE);
    imshow("Correlation", templ);


    moveWindow("MyVideo", 0,0);
    //moveWindow("MyVideo1", 0,480);
    //moveWindow("MyVideo2", 640,0);
    //moveWindow("MyVideo3", 640,300);
    //moveWindow("MyVideo4", 640,600);


	while (1)
    {
		// read a new frame from video
        bool bSuccess = cap.read(resized_frame[0]);

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

        Point p = Point(-1, -1);
        Point finalP;
        float max_intensity = 0;
        float max_index = -1;
        float temp;
        for (int i = 0; i < 5; i++) {
            p = Point(-1, -1);
            correlation(resized_frame[i], templ, corr_mat, p, temp);
            if (temp > max_intensity) {
                max_intensity = temp;
                max_index = i;
                finalP = p;
            }
        }
        if (max_index != -1){
            double factor = pow(0.75, max_index);
            addBoundingBox(resized_frame[0], Point(finalP.x / factor, finalP.y / factor) , templ.cols / factor, templ.rows / factor);
        }
		imshow("MyVideo", resized_frame[0]);
		//imshow("MyVideo1", resized_frame[1]);
		//imshow("MyVideo2", resized_frame[2]);
		//imshow("MyVideo3", resized_frame[3]);
		//imshow("MyVideo4", resized_frame[4]);
		//imshow("Red Video", red);
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
            if (temp > maxIntensity && temp > 0.6) {
                maxIntensity = temp;
                max = Point(x, y);
            }
        }
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

