/**
	CS585_Assignment4.cpp
	@author: 
	@version: 

	CS585 Image and Video Computing Fall 2014
	Assignment 4: BATS
*/

#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

/**
	Reads the label maps contained in the text file and converts the label maps
	into a labelled matrix, as well as a binary image containing the segmented objects

	@param filename The filename string
	@param labelled The matrix that will contain the object label information
		Background is labelled 0, and object labels start at 1
	@param binary The matrix that will contain the segmented objects that can be displayed
		Background is labelled 0, and objects are labelled 255
*/
void convertFileToMat(String filename, Mat& labelled, Mat& binary);

/**
	Reads the x-y object detections from the text file and draws in on a black and white image

	@param filename The filename string
	@param binary3channel The 3 channel image corresponding to the 1 channel binary image outputted
		by convertFileToMat. The object detection positions will be drawn on this image
*/
void drawObjectDetections(String filename, Mat& binary3channel);

int main()
{
	String filename_seg = "Segmentation_Bats/CS585Bats-Segmentation_frame000000750.txt";
	Mat labelled, binary;
	convertFileToMat(filename_seg, labelled, binary);

	String filename_det = "Localization_Bats/CS585Bats-Localization_frame000000750.txt";
	Mat binary3C;
	cvtColor(binary, binary3C, CV_GRAY2BGR);
	drawObjectDetections(filename_det, binary3C);

	namedWindow( "Labelled", 1 );
	imshow("Labelled", labelled);
	namedWindow( "Binary", 1 );
	imshow("Binary", binary);
	namedWindow( "Detections", 1 );
	imshow("Detections", binary3C);

	char key = waitKey(0);
	return 0;
}

void convertFileToMat(String filename, Mat& labelled, Mat& binary)
{
	//read file
	ifstream infile(filename);
	vector <vector <int>> data;
	if (!infile){
		cout << "Error reading file!";
		return;
	}

	//read the comma separated values into a vector of vector of ints
	while (infile)
	{
		string s;
		if (!getline( infile, s )) break;

		istringstream ss(s);
		vector <int> datarow;

		while (ss)
		{
		  string srow;
		  int sint;
		  if (!getline( ss, srow, ',' )) break;
		  sint = atoi(srow.c_str()); //convert string to int
		  datarow.push_back(sint);
		}
		data.push_back(datarow);
	}

	//construct the labelled matrix from the vector of vector of ints
	labelled = Mat::zeros(data.size(), data.at(0).size(), CV_8UC1);
	for (int i = 0; i < labelled.rows; ++i)
		for (int j = 0; j < labelled.cols; ++j)
			labelled.at<uchar>(i,j) = data.at(i).at(j);
	
	//construct the binary matrix from the labelled matrix
	binary = Mat::zeros(labelled.rows, labelled.cols, CV_8UC1);
	for (int i = 0; i < labelled.rows; ++i)
		for (int j = 0; j < labelled.cols; ++j)
			binary.at<uchar>(i,j) = (labelled.at<uchar>(i,j) == 0) ? 0: 255;

}

void drawObjectDetections(String filename, Mat& binary3channel)
{
	ifstream infile(filename);
	vector <vector <int>> data;
	if (!infile){
		cout << "Error reading file!";
		return;
	}

	//read the comma separated values into a vector of vector of ints
	while (infile)
	{
		string s;
		if (!getline( infile, s )) break;

		istringstream ss(s);
		vector <int> datarow;

		while (ss)
		{
		  string srow;
		  int sint;
		  if (!getline( ss, srow, ',' )) break;
		  sint = atoi(srow.c_str()); //convert string to int
		  datarow.push_back(sint);
		}
		data.push_back(datarow);
	}

	//draw red circles on the image
	for (int i = 0; i < data.size(); ++i)
	{
		Point center(data.at(i).at(0), data.at(i).at(1)) ;
		circle(binary3channel, center, 3, Scalar(0,0,255), -1, 8);
	}
		

}