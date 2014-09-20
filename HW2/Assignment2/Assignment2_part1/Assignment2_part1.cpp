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
	namedWindow("MyVideo",WINDOW_AUTOSIZE);
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
		imshow("MyVideo", frame);

		if (waitKey(30) == 27) 
		{
			cout << "esc key is pressed by user" << endl;
			break; 
		}

	}
	
	cap.release();
	return 0;
}

