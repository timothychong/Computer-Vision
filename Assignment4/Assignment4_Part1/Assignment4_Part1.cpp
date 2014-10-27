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


void convertFileToMat(String filename, Mat& labelled, Mat& binary);
void drawObjectDetections(vector<FilterBundle> data, Mat& binary3channel);
void centroidsFromFile(String filename, vector<vector<int>> & data);
void getLabelMatrix(String file, Mat & labelled);
void convertFileToMatWithColor(Mat & labelled, Mat& output, vector<FilterBundle> & bundles);
void printMat(Mat & mat);
void getHashMapArea(Mat & label, map<int,int> & area);

// struct use for easy comparison
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
    const int number_of_frames = 151;
    const int start_frame = 750;
	
	// get the proper name of the original sequences, ie get correct leading zeros
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
    //allocate the global data for all objects
    vector<FilterBundle> globalBundles;
    vector<vector<int>> initialVec;
    centroidsFromFile(files[0], initialVec);

    for(int a = 0; a < initialVec.size(); a++) {
        globalBundles.push_back(FilterBundle(Point(initialVec[a][0],initialVec[a][1])));
    }
    
	// clock variables:
	clock_t t1,t2;

	// run for each frame i
    for (int i = 1; i < files.size(); i++) {

		t1=clock();

        vector<vector<int>> vec;
        centroidsFromFile(files[i], vec);

        vector<int> tempGlobal;
        for (int a = 0; a < globalBundles.size(); a++ ){
            tempGlobal.push_back(a);
        }
        vector <int> usedDots;

        Mat labelled;
        getLabelMatrix(binary_files[i], labelled);
        map<int, int> area;
        getHashMapArea(labelled, area);

        //For every dots in the previous state
        for (int j = 0; j < globalBundles.size(); j++ ) {
            Point previousPrediction = globalBundles[j].getCurrentPrediction();
            //Make a prediction of the new dot
            Point prediction = globalBundles[j].predict();
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
                    float limit = 100;
                    if ( area.find(k + 1) != area.end() ) {
                        limit =  sqrt(area[k + 1]);
                        if (limit > 100) {
                            cout << limit << endl;
                        }
                    }else
                        cout << "Not found" << endl;
                    //Calculate distance and update shortest distance
                    float distance = sqrt( (prediction.x - location.x) * (prediction.x - location.x) + (prediction.y - location.y) * (prediction.y - location.y));

                    if (distance < limit && distance < shortestDistance){
                    //if (distance < shortestDistance){
                        shortestDistance = distance;
                        shortestIndex = k;
                    }
            }
            //If a shortest distance is found
            if (shortestIndex != -1){
                Point measurement = Point(vec[shortestIndex][0], vec[shortestIndex][1]);
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
            convertFileToMatWithColor(labelled, output, globalBundles);

	    	// draw the centroids
            drawObjectDetections(globalBundles, output);
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
			// printout to terminal
            cout << "Finished - " << i << ", time: " << t2-t1 << " ms" << endl;

        }
        int i = 0;
        cin >> i;
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

    void getLabelMatrix(String filename, Mat & labelled) {

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
    }
	
	// calculate and use area of object
    void getHashMapArea(Mat & labelled, map<int,int> & area) {
        for (int i = 0; i < labelled.rows; ++i)
            for (int j = 0; j < labelled.cols; ++j){
                int num = labelled.at<uchar>(i,j);
                if(num != 0) {
                    if ( area.find(num) == area.end() ) {
                      // not found
                      area[num] = 1;
                      } else {
                        // found
                        area[num] = area[num] + 1;
                    }
                }
            }
    }

    void convertFileToMatWithColor(Mat & labelled, Mat& output, vector<FilterBundle> & bundles)
    {
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
                    if(colorMap.find(labelled.at<uchar>(i,j)) != colorMap.end()){
                        Scalar color = colorMap[labelled.at<uchar>(i,j)];
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

// read in centroid data from file
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

// draw the centroids and tails for each object of the given frame
void drawObjectDetections(vector<FilterBundle> data, Mat& binary3channel)
{
    //draw the tails and centroids on each object
    for (int i = 0; i < data.size(); ++i)
    {
        Point center = data[i].getCurrentPrediction();
        // draw centroid circle
        circle(binary3channel, center, 2, Scalar(255,255,255), -1, 8);
        Point currPred = data[i].getCurrentPrediction();
        vector<Point> points = data[i].getPreviousLocations();
        for (int j = points.size() - 1; j >= 0; j--) {
            line(binary3channel, currPred, points[j], Scalar(255,255, 255), 1, CV_AA);
            currPred = points[j];
        }
    }
}

