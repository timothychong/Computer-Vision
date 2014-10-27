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

#define width 449
#define height 450

void readImages(String str, String extend, vector<String> &fileVector, int start_frame, int number_of_frames, int zeros);
void semgent2binary(vector<Mat> & images, vector<Mat> & segmented, vector<vector<Point>> & centroids);
void convertFileToMatWithColor(Mat & labelled, Mat& output, vector<FilterBundle> & bundles);
void drawObjectDetections(vector<FilterBundle> data, Mat& binary3channel);

// struct use for easy comparison
struct greater
{
    template<class T>
        bool operator()(T const &a, T const &b) const { return a > b; }
        };

int main()
{

	// first, determine number of frames to read in based on sequence we are talking about
	int start_frame,number_of_frames, zeros;
	String baseName, extend;
	
	// information related to iamge sequence we are dealing with
	start_frame = 1;
    number_of_frames = 53;
	//number_of_frames = 10-1;
	zeros = 2; // it ranges from frame 01 to 53, so there are 2 digits
	baseName = "Segmentation_Aqua/2014_aq_segmented ";
	extend = ".jpg";

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
	
	/*
	// block of code to save the binary images, for reference
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
        imwrite(fileo, output); }
		*/
		
        // now run the tracking process
        vector<FilterBundle> globalBundles;
        for(int a = 0; a < centroids[0].size(); a++) {
            globalBundles.push_back(FilterBundle(centroids[0][a]));
        }
	
    for(int i = 1; i < centroids.size(); i++) {

        vector<int> tempGlobal;
        for (int a = 0; a < globalBundles.size(); a++ ){
            tempGlobal.push_back(a);
        }
        vector <int> usedDots;

        for (int j = 0; j < globalBundles.size(); j++ ) {
            Point previousPrediction = globalBundles[j].getCurrentPrediction();
            //Make a prediction of the new dot
            Point prediction = globalBundles[j].predict();
            //If the prediction if out of bound, don't look for new dot
            if (prediction.x >= width || prediction.y >= height || prediction.x < 0 || prediction.y < 0){
                //cout << "Object going out of bound" << endl;
                continue;
            }

            float shortestDistance = numeric_limits<float>::max();
            float shortestIndex = -1;

            //Loop through every single new dot
            for (int k = 0; k < centroids[i].size(); k++) {
                    //See if the dot has already been taken
                    if(find(usedDots.begin(), usedDots.end(), k) != usedDots.end()) {
                        continue;
                    }
                    //Get the new dot's location
                    Point location = centroids[i][k];

                    //Calculate distance and update shortest distance
                    float distance = sqrt( (prediction.x - location.x) * (prediction.x - location.x) + (prediction.y - location.y) * (prediction.y - location.y));
                    if (distance < 20 && distance < shortestDistance){
                    //if (distance < shortestDistance){
                        shortestDistance = distance;
                        shortestIndex = k;
                    }
            }
            //If a shortest distance is found
            if (shortestIndex != -1){
                Point measurement = centroids[i][shortestIndex];
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
            for (int k = 0; k < centroids[i].size(); k++) {
                if(find(usedDots.begin(), usedDots.end(), k) != usedDots.end()) {
                    continue;
                }
                globalBundles.push_back(FilterBundle(centroids[i][k]));
            }
            for (unsigned int a = 0; a < globalBundles.size(); a++ ) {
                //cout << globalBundles[a] << endl;
            }

            Mat output;
            output = Mat::zeros(width, height, CV_8UC3);
            convertFileToMatWithColor(segmented[i], output, globalBundles);

            // draw the centroids and write the output
            drawObjectDetections(globalBundles, output);
            int number_of_digits = 3;
			int digit = (i == 0)? 2 : number_of_digits - 1 - (int)log10(i);
			ostringstream vv;
			vv << i;
			String fileo = "Tracked_fish/fish_";
			fileo.append(string(digit, '0'));
			fileo.append(string(vv.str()));
			fileo.append(string(".jpg"));
            imwrite(fileo, output);
            imwrite(fileo, output);
        }

	char key = waitKey(0);

	return 0;
}

// converts the segmented objects to the according object track color
void convertFileToMatWithColor(Mat & labelled, Mat& output, vector<FilterBundle> & bundles)
{

    map<int, Scalar> colorMap;
    for(int a = 0; a < bundles.size(); a++ ) {
        int p = bundles[a].currentIndex;
        colorMap[p] = bundles[a].color;
    }

    //construct the binary matrix from the labelled matrix
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

// draws the centroid and tails
void drawObjectDetections(vector<FilterBundle> data, Mat& binary3channel)
{
    //draw red circles on the image
    for (int i = 0; i < data.size(); ++i)
    {
        Point center = data[i].getCurrentPrediction();
		// draw the centroid
        circle(binary3channel, center, 2, Scalar(255,255,255), -1, 8);
        Point currPred = data[i].getCurrentPrediction();
        vector<Point> points = data[i].getPreviousLocations();
        for (int j = points.size() - 1; j >= 0; j--) {
        	// make the tails just white
            Scalar s = Scalar(255,255,255);
            line(binary3channel, currPred, points[j], s, 1, CV_AA);
            currPred = points[j];
        }
    }
}

// function for reading frames into a vector
void readImages(String str, String extend, vector<String> &fileVector, int start_frame, int number_of_frames, int zeros){
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

// runs the segmentation of the image from source
void semgent2binary(vector<Mat> & images, vector<Mat> & segmented, vector<vector<Point>> & centroids){

	for (int im=0; im<images.size(); im++){
		//namedWindow("window");
		imshow("original",images[im]);
		Mat binary = Mat::zeros(images[im].size(),CV_8UC1);

		// first color red parts to white, and everything else stays black
		for (int x=0; x< images[im].rows; x++){
			for (int y=0; y< images[im].cols; y++){
				Vec3b color = images[im].at<Vec3b>(x,y);
				// thresholding to get only red of fish, and not any of the green
				if (images[im].at<Vec3b>(x,y)[2] > 70 && images[im].at<Vec3b>(x,y)[1] < 120){
					binary.at<uchar>(x,y) = 255;
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

		// now run contours (fill with according color), this is the FIRST contours run.
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		Mat contour_output = Mat::zeros(binary.size(), CV_8UC1);

		// find and fill each contour with an increasing number
		findContours(binary,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
		for (int i=0; i< contours.size(); i++){
			drawContours(contour_output, contours, i, 255, CV_FILLED, 8, hierarchy);
		}
		//cout << "no. first contours: " << contours.size() << endl;

		// run contours AGAIN to clean up geometry and other conflicts
		// (e.g. before had 79 contours, after 38, there were overlaps the first time)
		Mat finalSegmented = Mat::zeros(binary.size(),CV_8UC1);
		findContours(contour_output,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
		//cout << "no. second contours: " << contours.size() << endl;
		vector<Point> centroidVec;
        int a = 0;
		for (int i=0; i< contours.size(); i++){

			Moments m = moments(contours[i], true);
			Point centroid(m.m10/m.m00,m.m01/m.m00);
            if (centroid.x > 0 && centroid.x < width && centroid.y > 0 && centroid.y < height) {
                centroidVec.push_back(centroid);
                // fill the object with the according color
                //drawContours(contour_output, contours, i, i*2+50, CV_FILLED, 8, hierarchy);
                drawContours(finalSegmented, contours, i, a + 1, CV_FILLED, 8, hierarchy);
                a++;
            }
		}

		centroids.push_back(centroidVec);
		segmented.push_back(finalSegmented.clone());

	}

	// all done!
	cout << "segmented image" << endl;
}
