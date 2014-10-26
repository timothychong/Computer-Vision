#ifndef FILTER_BUNDLE_H
#define FILTER_BUNDLE_H

#include <vector>
using namespace cv;
#define vx_init 0
#define vy_init 0
#define tail_size 4

#include "opencv2/core/core.hpp"
#include "stdafx.h"
#include "filterBundle.h"
#include <iostream>
#include <ctime>
#include <cstdlib>
using namespace std;

void printMat1(Mat & mat) {
    for (int i = 0; i < mat.rows ; i++ ) {
        for (int j = 0; j < mat.cols ; j++ ) {
            cout << mat.at<float>(i,j) << " ";
        }
        cout << endl;
    }
}

class FilterBundle
{
private:
    vector<Point> previousLocations;
    Point currentPrediction;
    KalmanFilter KF;
public:
    Scalar color;
    int currentIndex;
    FilterBundle(Point origLoc){
        KF.init(4, 2, 0);
        KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
        Mat a = KF.transitionMatrix;
        setIdentity(KF.measurementMatrix);
        setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-4));
        setIdentity(KF.errorCovPost, Scalar::all(.1));
        KF.statePre.at<float>(0) = origLoc.x;
        KF.statePre.at<float>(1) = origLoc.y;
        KF.statePre.at<float>(2) = vx_init;
        KF.statePre.at<float>(3) = vy_init;
        currentPrediction = origLoc;
        update(origLoc);
        color[0] = rand() % 256;
        color[1] = rand() % 256;
        color[2] = rand() % 256;
    }

    Point predict ()
    {
        Mat predictionMat = KF.predict();
        return Point(predictionMat.at<float>(0), predictionMat.at<float>(1));
    }
    Point update (Point p){
        Mat estimated = KF.correct(*(Mat_<float>(2, 1) << p.x, p.y));
        if (previousLocations.size() == tail_size){
            previousLocations.erase(previousLocations.begin());
        }
        previousLocations.push_back(currentPrediction);
        currentPrediction = Point(estimated.at<float>(0), estimated.at<float>(1));
        return currentPrediction;
    }
    vector<Point> getPreviousLocations(){
        return previousLocations;
    }
    Point getCurrentPrediction(){
        return currentPrediction;
    }
};

ostream& operator<< (ostream& os, FilterBundle & fb)
{
    Point p = fb.getCurrentPrediction();
    os << "FilterBundle at " << p.x << " , " << p.y << ". Previous: ";
    vector<Point> pred = fb.getPreviousLocations();
    if (pred.size() == 0) {
        os << "None";
        return os;
    }
    for( int i = 0; i < pred.size(); i++) {
        os << pred[i].x <<"," << pred[i].y << "; ";
    }
    return os;
}

#endif
