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
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>
#include "filterBundle.h"
#include <limits>
#include <time.h>


using namespace cv;
using namespace std;

#define width 1024
#define height 1024

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
void drawObjectDetections(vector<FilterBundle> data, Mat& binary3channel);
void centroidsFromFile(String filename, vector<vector<int>> & data);
void convertFileToMatWithColor(String filename, Mat& output, vector<FilterBundle> & bundles);
void printMat(Mat & mat);

struct greater
{
    template<class T>
        bool operator()(T const &a, T const &b) const { return a > b; }
        };

int main()
{
    srand(time(0));

    vector <String> files, binary_files;
    const int number_of_digits = 3;
    const int number_of_frames = 20;
    const int start_frame = 750;

    for ( int i= start_frame; i <  start_frame + number_of_frames; i ++ ) {
        String str = "Localization_Bats/CS585Bats-Localization_frame000000";
        String str_binary = "Segmentation_Bats/CS585Bats-Segmentation_frame000000";
        int digit = (i == 0)? 2 : number_of_digits - 1 - (int)log10(i);

        ostringstream ss;
        ss << i;

        str.append(string(digit, '0'));
        str_binary.append(string(digit, '0'));
        str.append(string(ss.str()));
        str_binary.append(string(ss.str()));
        str.append(".txt");
        str_binary.append(".txt");

        files.push_back(str);
        binary_files.push_back(str_binary);
    }
    vector<FilterBundle> globalBundles;

    vector<vector<int>> initialVec;
    centroidsFromFile(files[0], initialVec);

    for(int a = 0; a < initialVec.size(); a++) {
        globalBundles.push_back(FilterBundle(Point(initialVec[a][0],initialVec[a][1])));
    }
    //Mat labelled, binary;
    //convertFileToMat(binary_files[0], labelled, binary);
    //cout << binary.rows << " " << binary.cols << endl;

	// clock variables:
	clock_t t1,t2;

    for (int i = 1; i < files.size(); i++) {
        //Mat labelled, binary;
        //convertFileToMat(binary_files[i], labelled, binary);
        //imshow(str, binary);

		// timing for clock:
		t1=clock();

        vector<vector<int>> vec;
        centroidsFromFile(files[i], vec);

        vector<int> tempGlobal;
        for (int a = 0; a < globalBundles.size(); a++ ){
            tempGlobal.push_back(a);
        }
        vector <int> usedDots;

        //For every dots in the previous state
        for (int j = 0; j < globalBundles.size(); j++ ) {
            Point previousPrediction = globalBundles[j].getCurrentPrediction();
            //Make a prediction of the new dot
            Point prediction = globalBundles[j].predict();
            //cout << "Working on " << previousPrediction.x << " " << previousPrediction.y << " Predict to be: " << prediction.x << " " << prediction.y << endl;
            //If the prediction if out of bound. Don't look for new dot
            if (prediction.x >= width || prediction.y >= height || prediction.x < 0 || prediction.y < 0){
                //cout << "Object going out of bound" << endl;
                continue;
            }

            float shortestDistance = numeric_limits<float>::max();
            float shortestIndex = -1;

            //Loop through every single new dot
            for (int k = 0; k < vec.size(); k++) {
                    //See if the dot has already been taken
                    if(find(usedDots.begin(), usedDots.end(), k) != usedDots.end()) {
                        continue;
                    }
                    //Get the new dot's location
                    Point location = Point(vec[k][0], vec[k][1]);

                    //Calculate distance and update shortest distance
                    float distance = sqrt( (prediction.x - location.x) * (prediction.x - location.x) + (prediction.y - location.y) * (prediction.y - location.y));
                    //if (distance < 2000 && distance < shortestDistance){
                    if (distance < shortestDistance){
                        shortestDistance = distance;
                        shortestIndex = k;
                    }
            }
            //If a shortest distance is found
            if (shortestIndex != -1){
                Point measurement = Point(vec[shortestIndex][0], vec[shortestIndex][1]);
                //cout << "Updating: " << previousPrediction.x << " " << previousPrediction.y << " to " << measurement.x << " "  << measurement.y << endl;;
                //Take it out from the leftovers (to be deleted)
                tempGlobal.erase(remove(tempGlobal.begin(), tempGlobal.end(), j), tempGlobal.end());
                globalBundles[j].currentIndex = shortestIndex + 1;
                Point estimation = globalBundles[j].update(measurement);
                usedDots.push_back(shortestIndex);
            }
            //} else
                    //cout << "Can't find shortest distance" << endl;
        }
            //Sort that so that deletion won't be messed up
            sort(tempGlobal.begin(), tempGlobal.end(), greater());

            //Removing left overs
            for (int a = 0; a < tempGlobal.size(); a++) {
                globalBundles.erase(globalBundles.begin() + tempGlobal[a]);
            }
            //Adding new bodies
            for (int k = 0; k < vec.size(); k++) {
                if(find(usedDots.begin(), usedDots.end(), k) != usedDots.end()) {
                    continue;
                }
                globalBundles.push_back(FilterBundle(Point(vec[k][0],vec[k][1])));
            }
            for (unsigned int a = 0; a < globalBundles.size(); a++ ) {
                //cout << globalBundles[a] << endl;
            }

            Mat output;
            convertFileToMatWithColor(binary_files[i], output, globalBundles);

			// draw the centroids
			/*
			for (int j=0; j<centroids.size(); j++){
				circle(output, centroids[j], 2, Scalar(0,0,255), -1, 8);
			}
			*/

			/* draw the tails! */

			// end timing
			t2=clock();
			// write out the image file
			String fileo = "Tracked_Bats/trackBat_";

			int digit = (i == 0)? 2 : number_of_digits - 1 - (int)log10(i);
			ostringstream vv;
			vv << i;

			fileo.append(string(digit, '0'));
			fileo.append(string(vv.str()));
			fileo.append(string(".jpg"));
            imwrite(fileo, output);
			//imshow("window", output);
			// printout to terminal
            cout << "Finished - " << i << ", time: " << t2-t1 << " ms" << endl;

        }
		//char key = waitKey(0);
        return 0;
    }

    void printMat(Mat & mat) {
        for (int i = 0; i < mat.rows ; i++ ) {
            for (int j = 0; j < mat.cols ; j++ ) {
                cout << mat.at<float>(i,j) << " ";
            }
            cout << endl;
        }
    }

    void convertFileToMatWithColor(String filename, Mat& output, vector<FilterBundle> & bundles)
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
        Mat labelled = Mat::zeros(data.size(), data.at(0).size(), CV_8UC1);

        for (int i = 0; i < labelled.rows; ++i)
            for (int j = 0; j < labelled.cols; ++j)
                labelled.at<uchar>(i,j) = data.at(i).at(j);

        map<int, Scalar> colorMap;
        for(int a = 0; a < bundles.size(); a++ ) {
            int p = bundles[a].currentIndex;
            colorMap[p] = bundles[a].color;
        }


        //construct the binary matrix from the labelled matrix
        output = Mat::zeros(labelled.rows, labelled.cols, CV_8UC3);
        for (int i = 0; i < labelled.rows; ++i)
            for (int j = 0; j < labelled.cols; ++j){
                if (labelled.at<uchar>(i,j) != 0){
                    if(colorMap.find(data.at(i).at(j)) != colorMap.end()){
                        Scalar color = colorMap[data.at(i).at(j)];
                        output.at<Vec3b>(i,j)[0] = color[0];
                        output.at<Vec3b>(i,j)[1] = color[1];
                        output.at<Vec3b>(i,j)[2] = color[2];
                    } else {
                        output.at<Vec3b>(i,j)[0] = 255;
                        output.at<Vec3b>(i,j)[1] = 255;
                        output.at<Vec3b>(i,j)[2] = 255;
                }
            }
        }
}

void centroidsFromFile(String filename, vector<vector<int>> & data) {
	ifstream infile(filename);
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
}

void drawObjectDetections(vector<FilterBundle> data, Mat& binary3channel)
{
    //draw red circles on the image
    for (int i = 0; i < data.size(); ++i)
    {
        Point center = data[i].getCurrentPrediction();
        //cout << data[i].color<< endl;
        circle(binary3channel, center, 3, data[i].color, -1, 8);
    }
}

