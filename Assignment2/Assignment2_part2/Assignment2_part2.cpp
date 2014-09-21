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
	PART B
*/

#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>


using namespace cv;
using namespace std;

void mySkinDetect(Mat& src, Mat& dst);
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);
void myMotionEnergy(Vector<Mat> mh, Mat& dst);
int myMax(int a, int b, int c);
int myMin(int a, int b, int c);
void correlation(Mat & src, Mat & templ, Mat & dest, Point & max, float & maxIntensity);


int main()
{
	VideoCapture cap(0);
	// reading image sequence for hand position,
	// was going to have animated sequence, but was running oo slow.
	//VideoCapture ani;
	//ani.open("animation/animation_%04d.png");

	// if not successful, exit program
    if (!cap.isOpened())
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }

	Mat frame;
    Mat dst;
	namedWindow("MyVideo",WINDOW_AUTOSIZE);
	namedWindow("Result",WINDOW_AUTOSIZE);
	moveWindow("MyVideo", 0,200);
	moveWindow("Result",660,200);

	Mat ani_frame = imread("animation/animation_0005.png", CV_LOAD_IMAGE_COLOR);
    
	//Templates
    Mat paper = imread("template_paper.PNG", CV_LOAD_IMAGE_COLOR);
    Mat paper_binary = Mat::zeros(paper.rows, paper.cols, CV_8UC1);
    mySkinDetect(paper, paper_binary);

    Mat rock = imread("template_rock.PNG", CV_LOAD_IMAGE_COLOR);
    Mat rock_binary = Mat::zeros(rock.rows, rock.cols, CV_8UC1);
    mySkinDetect(rock, rock_binary);

    Mat scissors = imread("template_scissors.PNG", CV_LOAD_IMAGE_COLOR);
    Mat scissors_binary = Mat::zeros(scissors.rows, scissors.cols, CV_8UC1);
    mySkinDetect(scissors, scissors_binary);

	// template vector, to easily add more or swap different hand shapes
    vector<Mat> templates;
    templates.push_back(paper_binary);
    templates.push_back(scissors_binary);
    templates.push_back(rock_binary);

	// debug testing colors
    vector<Vec3b> colors = vector<Vec3b>();
    Vec3b red = Vec3b(0,0,255);
    Vec3b blue = Vec3b(255,0,0);
    Vec3b green = Vec3b(0,255,0);
    colors.push_back(red);
    colors.push_back(blue);
    colors.push_back(green);
	
	// veriables related to GUI and feedback
	int watchdog =0;
	int state=0;
	int animation_frame;
	int animation_set;
	int frequency = 0;

	// this is used for saying which gesture was last active.
	// -1 no shape, 0 = paper, 1=Scissors, 2=Rock
	float finalIndex = -1;

	// loop for every frame of the video
	while (1)
    {
		// this level runs every frame
		
		// read a new frame from video
		bool bSuccess = cap.read(frame);
		//if not successful, break loop
		if (!bSuccess)
		{
			 cout << "Cannot read a frame from video stream" << endl;
			 break;
		}

		if (watchdog >= frequency){
			// this level only runs ever n frames
			watchdog = 0;
			
			dst = Mat::zeros(frame.rows, frame.cols, CV_8UC1);

			cout << endl;
			mySkinDetect(frame,dst);
			Point p;
			float maxIntensity;
			Mat dest;
			float finalMax = 0;
			finalIndex = -1;
			for (int i = 0 ; i < 3; i++) {
				p = Point(-1,-1);
				maxIntensity = 0;
				correlation(dst, templates[i], dest, p,  maxIntensity);


				if ( i == 0 ) {
					maxIntensity *= 1.25; // correlation adjustment 1.8
					cout << "Paper Matching Correlation: \t" << maxIntensity << endl;

				} else if ( i == 1) {
					maxIntensity *= 1.2; // corelation adjustment 1.8
					cout << "Scissors Matching Correlation: \t" << maxIntensity << endl;

				} else if ( i == 2) {
					maxIntensity *= 1; // corelation adjustment 0.3
					cout << "Rock Matching Correlation: \t" << maxIntensity << endl;

				}

				if ( maxIntensity > finalMax && maxIntensity > 0.6) {
					finalMax = maxIntensity;
					finalIndex = i;
				}

			}
			cout << endl;

			//cout << i << " " << maxIntensity << endl;
			imshow("Result", dst);

			
		}

		// for testing debug colos, easy visualizaiton (change colo block fr matching)
		/*
		if (finalIndex != -1) {
			for (int i = 0; i < 100; i ++ ) {
				for (int j = 0; j < 100; j ++ ) {
					frame.at<Vec3b>(i, j) = colors[finalIndex];
				}
			}
		}
		*/

		// below runs every frame

		// overlay the frame of he different hand jestures on the output image.
		ani_frame.copyTo(frame(Rect(170,frame.rows-100,300,100)));//,ani_frame,mask_image);
		// showing which one was selected
		int tmp = finalIndex;
		if (finalIndex != -1){
			for (int x=0; x<100; x++){
				for (int y=0; y<100; y++){
					//Vec3b pixel = frame.at<Vec3b>(170+x*100,y);
					frame.at<Vec3b>(y+frame.rows-100,170+x+finalIndex*100)[0] = 150;
					//mask_image.at<uchar>(y, x+100*tmp) = 255;
				}
			}
		}

		watchdog++;
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

//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
    //Surveys of skin color modeling and detection techniques:
    //Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
    //Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            //For each pixel, compute the average intensity of the 3 color channels
            Vec3b intensity = src.at<Vec3b>(i,j); //Vec3b is a vector of 3 uchar (unsigned character)
            int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
            if ((R > 75 && G > 40 && B > 20) && (myMax(R,G,B) - myMin(R,G,B) > 15) && (abs(R-G) > 15) && (R > G) && (R > B)){
                dst.at<uchar>(i,j) = 255;
            }
			//if ((R > 95 && G > 40 && B > 20) && (myMax(R,G,B) - myMin(R,G,B) > 15) && (abs(R-G) > 15) && (R > G) && (R > B)){
            //    dst.at<uchar>(i,j) = 255;
            //}
        }
    }
}

//Function that does frame differencing between the current frame and the previous frame
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {
    //For more information on operation with arrays: http://docs.opencv.org/modules/core/doc/operations_on_arrays.html
    //For more information on how to use background subtraction methods: http://docs.opencv.org/trunk/doc/tutorials/video/background_subtraction/background_subtraction.html
    absdiff(prev, curr, dst);
    Mat gs = dst.clone();
    cvtColor(dst, gs, CV_BGR2GRAY);
    dst = gs > 50;
    Vec3b intensity = dst.at<Vec3b>(100,100);
}

//Function that accumulates the frame differences for a certain number of pairs of frames
void myMotionEnergy(Vector<Mat> mh, Mat& dst) {
    Mat mh0 = mh[0];
    Mat mh1 = mh[1];
    Mat mh2 = mh[2];

    for (int i = 0; i < dst.rows; i++){
        for (int j = 0; j < dst.cols; j++){
            if (mh0.at<uchar>(i,j) == 255 || mh1.at<uchar>(i,j) == 255 ||mh2.at<uchar>(i,j) == 255){
                dst.at<uchar>(i,j) = 255;
            }
        }
    }
}
//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
	int m = a;
    (void)((m < b) && (m = b));
    (void)((m < c) && (m = c));
     return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
	int m = a;
    (void)((m > b) && (m = b));
    (void)((m > c) && (m = c));
     return m;
}


void correlation(Mat & src, Mat & templ, Mat & dest, Point & max, float & maxIntensity) {
    matchTemplate( src, templ, dest, CV_TM_CCOEFF_NORMED);
    maxIntensity = 0;
    float temp = 0;
    for(int x = 0; x < dest.cols; x ++ ) {
        for(int y = 0; y < dest.rows; y ++ ) {
            temp = dest.at<float>(y, x);
            if (temp > maxIntensity && temp > 0.1) {
                maxIntensity = temp;
                max = Point(x, y);
            }
        }
    }
}
